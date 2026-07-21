"""Focused checks for the explicit public-mirror allowlists."""

import hashlib
import json

import pytest

from scripts import (
    build_anonymous_analysis_artifact,
    export_binary_seed_provenance,
    render_paper_tables,
    sync_repositories,
)


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_mock_public_seed_release(
    repo_root,
    monkeypatch,
    *,
    mock_analysis_validator=True,
    mock_provenance_validator=True,
    mock_scheduler_loader=True,
):
    digest = {letter: letter * 64 for letter in "abcdef"}
    analysis_cells = []
    provenance_cells = []
    for dataset_index in range(5):
        dataset = f"dataset-{dataset_index}"
        for model_index in range(2):
            model = f"model-{model_index}"
            condition = f"condition-{model_index}"
            sources = {}
            for seed in (1, 2):
                run_id = f"run-{dataset_index}-{model_index}-{seed}"
                manifest_sha256 = hashlib.sha256(
                    f"manifest-{run_id}".encode("ascii")
                ).hexdigest()
                records_sha256 = hashlib.sha256(
                    f"records-{run_id}".encode("ascii")
                ).hexdigest()
                sources[str(seed)] = {
                    "logical_id": f"seed-{seed}/{dataset}/{condition}/{run_id}",
                    "manifest_sha256": manifest_sha256,
                    "records_sha256": records_sha256,
                }
                provenance_cells.append(
                    {
                        "dataset_id": dataset,
                        "condition_id": condition,
                        "model_id": model,
                        "training_seed": seed,
                        "num_samples": 3,
                        "assembly": {
                            "run_id": run_id,
                            "manifest_sha256": manifest_sha256,
                            "records_sha256": records_sha256,
                        },
                    }
                )
            analysis_cells.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "model": model,
                    "num_images_per_seed": 3,
                    "sources": sources,
                }
            )
    analysis = {
        "schema_version": 1,
        "artifact_type": "selectseg.binary_seed_public_analysis",
        "analysis": {},
        "provenance": {
            "source_analysis_sha256": digest["a"],
            "downstream_lock_sha256": digest["b"],
            "canonical_seed0": {
                "analysis_sha256": digest["c"],
                "campaign_lock_sha256": digest["d"],
            },
            "analysis_source_sha256": digest["e"],
        },
        "cells": analysis_cells,
        "gate_c": {"fired": False},
    }
    scheduler = {
        "summary_schema_version": 1,
        "artifact_type": "selectseg.public_seed_scheduler_summary",
        "status": "complete",
    }
    analysis_payload = _json_bytes(analysis)
    provenance = {
        "schema_version": 1,
        "artifact_type": "selectseg.binary_seed_public_provenance",
        "campaign": {
            "downstream_lock_sha256": digest["b"],
            "canonical_seed0": {
                "analysis_sha256": digest["c"],
                "campaign_lock_sha256": digest["d"],
            },
        },
        "scheduler": scheduler,
        "cells": provenance_cells,
        "analysis": {
            "source_analysis_sha256": digest["a"],
            "portable_analysis_sha256": hashlib.sha256(analysis_payload).hexdigest(),
            "analysis_source_sha256": digest["e"],
            "cell_count": len(analysis["cells"]),
            "gate_c_fired": analysis["gate_c"]["fired"],
            "gate_c_sha256": sync_repositories._canonical_sha256(analysis["gate_c"]),
        },
    }
    values = {
        "seed_robustness_analysis.json": analysis,
        "seed_scheduler_summary.json": scheduler,
        "seed_provenance.json": provenance,
    }
    release_root = repo_root / "outputs" / "public_seed"
    release_root.mkdir(parents=True)
    for name, value in values.items():
        (release_root / name).write_bytes(_json_bytes(value))

    calls = []
    if mock_analysis_validator:
        monkeypatch.setattr(
            export_binary_seed_provenance,
            "_validate_public_analysis",
            lambda value: calls.append("analysis") or value,
        )
    if mock_provenance_validator:
        monkeypatch.setattr(
            export_binary_seed_provenance,
            "_validate_public_provenance",
            lambda value: calls.append("provenance") or value,
        )
    if mock_scheduler_loader:

        def validate_summary(value):
            calls.append("scheduler")
            return value

        monkeypatch.setattr(
            export_binary_seed_provenance,
            "_validate_scheduler_public_summary",
            validate_summary,
        )
    return values, calls


def test_generated_manifest_loader_rejects_duplicate_json_keys(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,"artifact_type":"selectseg.test"}\n'
    )

    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        sync_repositories._load_generated_manifest(
            manifest, artifact_type="selectseg.test"
        )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_generated_manifest_loader_rejects_nonfinite_json_constants(tmp_path, constant):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"artifact_type":"selectseg.test",'
        f'"invalid":{constant}}}\n'
    )

    with pytest.raises(RuntimeError, match="non-finite JSON constant"):
        sync_repositories._load_generated_manifest(
            manifest, artifact_type="selectseg.test"
        )


def test_final_table_allowlist_matches_schema_v2_renderer():
    expected = tuple(f"Tables/{name}" for name in render_paper_tables.OUTPUT_NAMES)
    assert sync_repositories.FINAL_GENERATED_TABLE_FILES == expected
    assert len(expected) == 6
    assert set(expected) <= set(sync_repositories.MANUSCRIPT_FILES)


def test_completion_sentinel_is_mirrored_with_final_tables():
    expected = f"Tables/{render_paper_tables.COMPLETION_MARKER}"
    assert sync_repositories.RESULTS_COMPLETION_SENTINEL_FILE == expected
    assert expected in set(sync_repositories.MANUSCRIPT_FILES)

    targets = sync_repositories._targets()
    overleaf_destinations = {
        item.destination.relative_to(targets["overleaf"].root).as_posix()
        for item in targets["overleaf"].items
    }
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    assert expected in overleaf_destinations
    assert f"docs/{expected}" in github_destinations


def test_explicit_manuscript_pdf_assets_are_visible_and_mirrored():
    expected = ("Figures/loss_indexed_framework.pdf",)
    assert sync_repositories.MANUSCRIPT_PDF_FILES == expected
    assert set(expected) <= set(sync_repositories.MANUSCRIPT_FILES)
    assert all(
        (sync_repositories.REPO_ROOT / "docs" / path).is_file() for path in expected
    )

    ignore_lines = set(
        (sync_repositories.REPO_ROOT / ".gitignore").read_text().splitlines()
    )
    assert "*.pdf" in ignore_lines
    assert "!docs/Figures/*.pdf" in ignore_lines

    targets = sync_repositories._targets()
    overleaf_destinations = {
        item.destination.relative_to(targets["overleaf"].root).as_posix()
        for item in targets["overleaf"].items
    }
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    for path in expected:
        assert path in overleaf_destinations
        assert f"docs/{path}" in github_destinations


def test_optional_risk_coverage_assets_are_all_or_none(tmp_path, monkeypatch):
    expected = (
        "Figures/risk_coverage_all_indexed_pet.pdf",
        "Figures/risk_coverage_all_indexed_kvasir.pdf",
        "Figures/risk_coverage_all_indexed_fives.pdf",
        "Figures/risk_coverage_all_indexed_isic.pdf",
        "Figures/risk_coverage_all_indexed_tn3k.pdf",
    )
    assert sync_repositories.OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES == expected
    sentinel = "Figures/risk_coverage_complete.tex"
    assert sync_repositories.OPTIONAL_GENERATED_FIGURE_SENTINEL_FILES == (sentinel,)
    manifest = "Figures/risk_coverage_manifest.json"
    assert sync_repositories.OPTIONAL_GENERATED_FIGURE_MANIFEST_FILES == (manifest,)

    docs_root = tmp_path / "docs"
    figure_root = docs_root / "Figures"
    figure_root.mkdir(parents=True)
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    empty_targets = sync_repositories._targets()
    overleaf_destinations = {
        item.destination.relative_to(empty_targets["overleaf"].root).as_posix()
        for item in empty_targets["overleaf"].items
    }
    github_destinations = {
        item.destination.relative_to(empty_targets["github"].root).as_posix()
        for item in empty_targets["github"].items
    }
    optional_assets = set(expected) | {sentinel, manifest}
    assert not optional_assets & overleaf_destinations
    assert not {f"docs/{path}" for path in optional_assets} & github_destinations

    (docs_root / expected[2]).write_bytes(b"%PDF-lone-generated-output")
    with pytest.raises(
        RuntimeError, match="risk-coverage generated group is incomplete"
    ):
        sync_repositories._targets()

    source_bundle = {"campaign_lock": {"sha256": "a" * 64}, "inputs": []}
    source_bundle_sha = sync_repositories._canonical_sha256(source_bundle)
    render_spec = {
        "source_artifact_bundle": source_bundle,
        "source_artifact_bundle_sha256": source_bundle_sha,
        "plot_source_sha256": "b" * 64,
        "outputs": [path.rsplit("/", 1)[-1] for path in expected],
    }
    render_spec_sha = sync_repositories._canonical_sha256(render_spec)
    output_records = []
    for path in expected:
        output = docs_root / path
        output.write_bytes(b"%PDF-complete-test-" + render_spec_sha.encode())
        output_records.append(
            {"path": output.name, "sha256": sync_repositories._sha256_file(output)}
        )
    render_manifest = {
        "schema_version": 1,
        "artifact_type": "selectseg.risk_coverage_render_manifest",
        "render_spec_sha256": render_spec_sha,
        "render_spec": render_spec,
        "outputs": output_records,
    }
    manifest_path = docs_root / manifest
    manifest_path.write_text(json.dumps(render_manifest, sort_keys=True) + "\n")
    manifest_sha = sync_repositories._sha256_file(manifest_path)
    (docs_root / sentinel).write_text(
        f"% {source_bundle_sha}\n% {render_spec_sha}\n% {manifest_sha}\n"
    )
    complete_targets = sync_repositories._targets()
    complete_overleaf_destinations = {
        item.destination.relative_to(complete_targets["overleaf"].root).as_posix()
        for item in complete_targets["overleaf"].items
    }
    complete_github_destinations = {
        item.destination.relative_to(complete_targets["github"].root).as_posix()
        for item in complete_targets["github"].items
    }
    assert optional_assets <= complete_overleaf_destinations
    assert {f"docs/{path}" for path in optional_assets} <= complete_github_destinations


def test_qualitative_generated_group_is_all_or_none(tmp_path, monkeypatch):
    figure_root = tmp_path / "docs" / "Figures"
    figure_root.mkdir(parents=True)
    (figure_root / "qualitative_pet.png").write_bytes(b"lone-panel")
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="qualitative generated group is incomplete"):
        sync_repositories._targets()


def test_qualitative_public_manifest_detects_any_tampered_panel(tmp_path, monkeypatch):
    docs_root = tmp_path / "docs"
    figure_root = docs_root / "Figures"
    figure_root.mkdir(parents=True)
    digests = {
        "selection_sha256": "a" * 64,
        "campaign_lock_sha256": "b" * 64,
        "renderer_source_sha256": "c" * 64,
        "source_render_manifest_sha256": "d" * 64,
    }
    tex = figure_root / "qualitative_cases.tex"
    tex.write_text("\n".join(digests.values()) + "\n")
    outputs = []
    for name in (
        "qualitative_pet.png",
        "qualitative_kvasir.png",
        "qualitative_fives.png",
        "qualitative_isic.png",
        "qualitative_tn3k.png",
    ):
        path = figure_root / name
        path.write_bytes(name.encode())
        outputs.append({"path": name, "sha256": sync_repositories._sha256_file(path)})
    outputs.append({"path": tex.name, "sha256": sync_repositories._sha256_file(tex)})
    manifest = {
        "schema_version": 1,
        "artifact_type": "selectseg.binary_qualitative_manuscript",
        "render_id": hashlib.sha256(
            (
                digests["selection_sha256"] + "\0" + digests["renderer_source_sha256"]
            ).encode("ascii")
        ).hexdigest()[:16],
        **digests,
        "outputs": outputs,
    }
    manifest_path = figure_root / "qualitative_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)
    targets = sync_repositories._targets()
    expected = set(sync_repositories.OPTIONAL_GENERATED_QUALITATIVE_FILES)
    overleaf_destinations = {
        item.destination.relative_to(targets["overleaf"].root).as_posix()
        for item in targets["overleaf"].items
    }
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    assert expected <= overleaf_destinations
    assert {f"docs/{path}" for path in expected} <= github_destinations

    sync_repositories._validate_qualitative_closure(docs_root)
    invalid_render_id = dict(manifest, render_id="0" * 16)
    manifest_path.write_text(json.dumps(invalid_render_id, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="render_id is inconsistent"):
        sync_repositories._validate_qualitative_closure(docs_root)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (figure_root / "qualitative_isic.png").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="generated output SHA-256 mismatch"):
        sync_repositories._validate_qualitative_closure(docs_root)


def test_optional_public_seed_results_are_exact_github_only_and_guard_selected(
    tmp_path, monkeypatch
):
    expected = (
        (
            "outputs/public_seed/seed_robustness_analysis.json",
            "results/seed_robustness_analysis.json",
        ),
        (
            "outputs/public_seed/seed_scheduler_summary.json",
            "results/seed_scheduler_summary.json",
        ),
        (
            "outputs/public_seed/seed_provenance.json",
            "results/seed_provenance.json",
        ),
    )
    assert sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES == expected
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    # No discovery: an unlisted neighbor does not activate the group.
    unexpected = tmp_path / "outputs" / "public_seed" / "private-ledger.jsonl"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("private\n", encoding="utf-8")
    empty_targets = sync_repositories._targets()
    empty_github = {
        item.destination.relative_to(empty_targets["github"].root).as_posix()
        for item in empty_targets["github"].items
    }
    assert not ({target for _, target in expected} & empty_github)

    # Analysis and scheduler summaries are legitimate intermediate products.
    # Without the write-last provenance guard, neither is publishable and
    # neither may make an unrelated Overleaf/GitHub sync fail.
    (tmp_path / expected[0][0]).write_text("{}\n", encoding="utf-8")
    (tmp_path / expected[1][0]).write_text("{}\n", encoding="utf-8")
    intermediate_targets = sync_repositories._targets()
    intermediate_github = {
        item.destination.relative_to(intermediate_targets["github"].root).as_posix()
        for item in intermediate_targets["github"].items
    }
    assert not ({target for _, target in expected} & intermediate_github)


def test_actual_public_seed_replay_is_complete_guarded_and_recomputable():
    root = sync_repositories.REPO_ROOT
    sync_repositories._validate_public_seed_replay_closure(root)
    selected = sync_repositories._validated_optional_public_seed_replay(root)
    assert selected == sync_repositories.OPTIONAL_PUBLIC_SEED_REPLAY_FILES
    assert len(selected) == 62
    assert selected[-2][0].endswith("seed_replay.lock.json")
    assert selected[-1][0].endswith("seed_replay.complete.json")


def test_public_seed_guard_present_with_missing_payloads_is_rejected(
    tmp_path, monkeypatch
):
    guard = tmp_path / sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[2][0]
    guard.parent.mkdir(parents=True)
    guard.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="public seed result group is incomplete"):
        sync_repositories._targets()


def test_guard_absent_seed_intermediates_preserve_existing_mirror_release(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    summary = source_root / sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[1][0]
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n", encoding="utf-8")
    assert sync_repositories._validated_optional_public_seed_results(source_root) == ()

    target_root = tmp_path / "target"
    expected = {}
    for _, destination in sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES:
        path = target_root / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"existing:{destination}".encode("utf-8")
        path.write_bytes(payload)
        expected[path] = payload

    target = sync_repositories.SyncTarget("github", target_root, ())
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )
    assert sync_repositories.sync(target, apply=True) == (0, 0)
    assert all(path.read_bytes() == payload for path, payload in expected.items())


def test_optional_public_seed_results_reject_symlink(tmp_path, monkeypatch):
    release_root = tmp_path / "outputs" / "public_seed"
    release_root.mkdir(parents=True)
    for source, _ in sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[:2]:
        (tmp_path / source).write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (tmp_path / sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[2][0]).symlink_to(
        outside
    )
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="incomplete or non-regular"):
        sync_repositories._targets()


def test_complete_public_seed_results_use_all_strict_validators_and_github_only(
    tmp_path, monkeypatch
):
    _, calls = _write_mock_public_seed_release(tmp_path, monkeypatch)
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    targets = sync_repositories._targets()
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    overleaf_destinations = {
        item.destination.relative_to(targets["overleaf"].root).as_posix()
        for item in targets["overleaf"].items
    }
    expected_destinations = {
        target for _, target in sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES
    }
    assert expected_destinations <= github_destinations
    assert not (expected_destinations & overleaf_destinations)
    assert calls == ["analysis", "scheduler", "provenance"]


@pytest.mark.parametrize(
    ("document", "validator", "message"),
    (
        ("seed_robustness_analysis.json", "analysis", "public seed analysis"),
        ("seed_provenance.json", "provenance", "public seed provenance"),
        (
            "seed_scheduler_summary.json",
            "scheduler",
            "public scheduler summary",
        ),
    ),
)
def test_public_seed_result_group_delegates_exact_schema_validation(
    tmp_path, monkeypatch, document, validator, message
):
    actual_analysis_validator = export_binary_seed_provenance._validate_public_analysis
    actual_provenance_validator = (
        export_binary_seed_provenance._validate_public_provenance
    )
    actual_scheduler_loader = (
        export_binary_seed_provenance._validate_scheduler_public_summary
    )
    values, _ = _write_mock_public_seed_release(tmp_path, monkeypatch)
    release_root = tmp_path / "outputs" / "public_seed"
    invalid = dict(values[document], unexpected_public_field=True)
    (release_root / document).write_bytes(_json_bytes(invalid))

    if validator == "analysis":
        monkeypatch.setattr(
            export_binary_seed_provenance,
            "_validate_public_analysis",
            actual_analysis_validator,
        )
    elif validator == "provenance":
        monkeypatch.setattr(
            export_binary_seed_provenance,
            "_validate_public_provenance",
            actual_provenance_validator,
        )
    else:
        monkeypatch.setattr(
            export_binary_seed_provenance,
            "_validate_scheduler_public_summary",
            actual_scheduler_loader,
        )
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)

    with pytest.raises((RuntimeError, ValueError), match=message):
        sync_repositories._targets()


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("analysis-bytes", "does not bind the analysis bytes"),
        ("scheduler", "embeds a different scheduler summary"),
        ("downstream", "analysis and provenance disagree"),
        ("gate", "analysis and provenance disagree"),
    ),
)
def test_public_seed_guard_rejects_cross_file_tampering(
    tmp_path, monkeypatch, tamper, message
):
    values, _ = _write_mock_public_seed_release(tmp_path, monkeypatch)
    release_root = tmp_path / "outputs" / "public_seed"
    analysis_path = release_root / "seed_robustness_analysis.json"
    provenance_path = release_root / "seed_provenance.json"

    if tamper == "analysis-bytes":
        analysis_path.write_bytes(analysis_path.read_bytes() + b"\n")
    elif tamper == "scheduler":
        summary = dict(values["seed_scheduler_summary.json"], status="changed")
        (release_root / "seed_scheduler_summary.json").write_bytes(_json_bytes(summary))
    elif tamper == "downstream":
        analysis = values["seed_robustness_analysis.json"]
        analysis["provenance"]["downstream_lock_sha256"] = "f" * 64
        analysis_payload = _json_bytes(analysis)
        analysis_path.write_bytes(analysis_payload)
        provenance = values["seed_provenance.json"]
        provenance["analysis"]["portable_analysis_sha256"] = hashlib.sha256(
            analysis_payload
        ).hexdigest()
        provenance_path.write_bytes(_json_bytes(provenance))
    else:
        provenance = values["seed_provenance.json"]
        provenance["analysis"]["gate_c_sha256"] = "f" * 64
        provenance_path.write_bytes(_json_bytes(provenance))

    monkeypatch.setattr(sync_repositories, "REPO_ROOT", tmp_path)
    with pytest.raises((RuntimeError, ValueError), match=message):
        sync_repositories._targets()


def test_validated_content_rejects_symlinked_source_ancestor(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "payload.txt").write_text("payload")
    (source_root / "linked").symlink_to(actual, target_is_directory=True)
    target_root = tmp_path / "target"
    target_root.mkdir()
    item = sync_repositories._item(
        source_root, "linked/payload.txt", target_root, "payload.txt"
    )

    with pytest.raises(RuntimeError, match="source ancestor is a symlink"):
        sync_repositories._validated_content(item, target_root.resolve())


def test_validated_content_rejects_source_outside_allowed_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("payload")
    target_root = tmp_path / "target"
    target_root.mkdir()
    item = sync_repositories.CopyItem(
        source_root=source_root,
        source=outside,
        destination=target_root / "outside.txt",
    )

    with pytest.raises(RuntimeError, match="source is outside its source root"):
        sync_repositories._validated_content(item, target_root.resolve())


def test_validated_content_rejects_symlinked_source_root(tmp_path):
    actual_root = tmp_path / "actual-source"
    actual_root.mkdir()
    (actual_root / "payload.txt").write_text("payload")
    source_root = tmp_path / "source"
    source_root.symlink_to(actual_root, target_is_directory=True)
    target_root = tmp_path / "target"
    target_root.mkdir()
    item = sync_repositories._item(
        source_root, "payload.txt", target_root, "payload.txt"
    )

    with pytest.raises(RuntimeError, match="source root is missing or a symlink"):
        sync_repositories._validated_content(item, target_root.resolve())


def test_validated_content_rejects_symlink_above_source_root(tmp_path):
    actual_parent = tmp_path / "actual-parent"
    source = actual_parent / "source" / "payload.txt"
    source.parent.mkdir(parents=True)
    source.write_text("payload")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)
    source_root = linked_parent / "source"
    target_root = tmp_path / "target"
    target_root.mkdir()
    item = sync_repositories._item(
        source_root, "payload.txt", target_root, "payload.txt"
    )

    with pytest.raises(RuntimeError, match="source ancestor is a symlink"):
        sync_repositories._validated_content(item, target_root.resolve())


def _publication_spec(name):
    return next(
        spec for spec in sync_repositories._publication_specs() if spec[0] == name
    )


def _publication_target(tmp_path, paths, *, include_unrelated=False):
    target_root = tmp_path / "target"
    target_root.mkdir()
    docs_root = sync_repositories.REPO_ROOT / "docs"
    items = []
    for path in paths:
        destination = target_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"stale:{path}".encode())
        items.append(sync_repositories._item(docs_root, path, target_root, path))
    if include_unrelated:
        destination = target_root / "README.md"
        destination.write_bytes(b"stale-unrelated")
        items.append(
            sync_repositories._item(
                sync_repositories.REPO_ROOT,
                "README.md",
                target_root,
                "README.md",
            )
        )
    return sync_repositories.SyncTarget("test", target_root, tuple(items))


def _public_seed_publication_target(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    _write_mock_public_seed_release(source_root, monkeypatch)
    monkeypatch.setattr(sync_repositories, "REPO_ROOT", source_root)
    target_root = tmp_path / "target"
    target_root.mkdir()
    items = []
    for source, destination in sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES:
        target_path = target_root / destination
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(f"stale:{destination}".encode("utf-8"))
        items.append(
            sync_repositories._item(source_root, source, target_root, destination)
        )
    return sync_repositories.SyncTarget("github", target_root, tuple(items))


def test_public_seed_publication_withdraws_guard_first_and_writes_it_last(
    tmp_path, monkeypatch
):
    target = _public_seed_publication_target(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )
    events = []
    original_withdraw = sync_repositories._withdraw_publication_guard
    original_atomic_write = sync_repositories._atomic_write

    def recording_withdraw(group, target_root):
        events.append(
            ("withdraw", group.guard.destination.relative_to(target.root).as_posix())
        )
        original_withdraw(group, target_root)

    def recording_atomic_write(destination, content, mode):
        events.append(("write", destination.relative_to(target.root).as_posix()))
        original_atomic_write(destination, content, mode)

    monkeypatch.setattr(
        sync_repositories, "_withdraw_publication_guard", recording_withdraw
    )
    monkeypatch.setattr(sync_repositories, "_atomic_write", recording_atomic_write)

    changed, unchanged = sync_repositories.sync(target, apply=True)

    expected = sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES
    assert (changed, unchanged) == (3, 0)
    assert events == [
        ("withdraw", expected[2][1]),
        ("write", expected[0][1]),
        ("write", expected[1][1]),
        ("write", expected[2][1]),
    ]


def test_public_seed_publication_group_rejects_non_github_target(tmp_path, monkeypatch):
    github_target = _public_seed_publication_target(tmp_path, monkeypatch)
    target = sync_repositories.SyncTarget(
        "overleaf", github_target.root, github_target.items
    )
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )

    with pytest.raises(RuntimeError, match="restricted to the GitHub mirror"):
        sync_repositories.sync(target, apply=True)


@pytest.mark.parametrize("failure_index", (0, 1, 2))
def test_interrupted_public_seed_publication_leaves_guard_absent(
    tmp_path, monkeypatch, failure_index
):
    target = _public_seed_publication_target(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )
    failure_destination = (
        target.root
        / (sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[failure_index][1])
    )
    guard_destination = (
        target.root / (sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[2][1])
    )
    original_atomic_write = sync_repositories._atomic_write

    def failing_atomic_write(destination, content, mode):
        if destination == failure_destination:
            raise OSError("injected seed publication failure")
        original_atomic_write(destination, content, mode)

    monkeypatch.setattr(sync_repositories, "_atomic_write", failing_atomic_write)

    with pytest.raises(OSError, match="injected seed publication failure"):
        sync_repositories.sync(target, apply=True)

    assert not guard_destination.exists()


def test_partial_public_seed_publication_never_withdraws_guard(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    _write_mock_public_seed_release(source_root, monkeypatch)
    target_root = tmp_path / "target"
    target_root.mkdir()
    guard_source, guard_target = sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES[2]
    guard_destination = target_root / guard_target
    guard_destination.parent.mkdir(parents=True)
    guard_destination.write_bytes(b"stale-guard")
    target = sync_repositories.SyncTarget(
        "github",
        target_root,
        (
            sync_repositories._item(
                source_root, guard_source, target_root, guard_target
            ),
        ),
    )
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )

    with pytest.raises(RuntimeError, match="publication group is incomplete"):
        sync_repositories.sync(target, apply=True)

    assert guard_destination.read_bytes() == b"stale-guard"


def test_apply_withdraws_guards_then_publishes_payload_manifest_guard(
    tmp_path, monkeypatch
):
    risk = _publication_spec("risk-coverage")
    qualitative = _publication_spec("qualitative")
    specs = (risk, qualitative)
    paths = tuple(
        path
        for _, payloads, manifest, guard, _ in specs
        for path in payloads + (manifest, guard)
    )
    target = _publication_target(tmp_path, paths, include_unrelated=True)
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )
    events = []
    original_withdraw = sync_repositories._withdraw_publication_guard
    original_atomic_write = sync_repositories._atomic_write

    def recording_withdraw(group, target_root):
        events.append(
            ("withdraw", group.guard.destination.relative_to(target.root).as_posix())
        )
        original_withdraw(group, target_root)

    def recording_atomic_write(destination, content, mode):
        events.append(("write", destination.relative_to(target.root).as_posix()))
        original_atomic_write(destination, content, mode)

    monkeypatch.setattr(
        sync_repositories, "_withdraw_publication_guard", recording_withdraw
    )
    monkeypatch.setattr(sync_repositories, "_atomic_write", recording_atomic_write)

    changed, unchanged = sync_repositories.sync(target, apply=True)

    assert changed == len(paths) + 1
    assert unchanged == 0
    assert events == (
        [("withdraw", spec[3]) for spec in specs]
        + [("write", path) for spec in specs for path in spec[1]]
        + [("write", spec[2]) for spec in specs]
        + [("write", spec[3]) for spec in specs]
        + [("write", "README.md")]
    )
    for path in paths:
        assert (target.root / path).read_bytes() == (
            sync_repositories.REPO_ROOT / "docs" / path
        ).read_bytes()
    assert (target.root / "README.md").read_bytes() == (
        sync_repositories.REPO_ROOT / "README.md"
    ).read_bytes()


@pytest.mark.parametrize(
    ("group_name", "failure_role"),
    (
        ("risk-coverage", "payload"),
        ("risk-coverage", "manifest"),
        ("risk-coverage", "guard"),
        ("qualitative", "payload"),
        ("qualitative", "manifest"),
        ("qualitative", "guard"),
    ),
)
def test_interrupted_generated_group_publish_leaves_guard_absent(
    tmp_path, monkeypatch, group_name, failure_role
):
    _, payloads, manifest, guard, _ = _publication_spec(group_name)
    paths = payloads + (manifest, guard)
    target = _publication_target(tmp_path, paths, include_unrelated=True)
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )
    failure_path = {
        "payload": payloads[0],
        "manifest": manifest,
        "guard": guard,
    }[failure_role]
    original_atomic_write = sync_repositories._atomic_write

    def failing_atomic_write(destination, content, mode):
        relative = destination.relative_to(target.root).as_posix()
        if relative == failure_path:
            raise OSError("injected publication failure")
        original_atomic_write(destination, content, mode)

    monkeypatch.setattr(sync_repositories, "_atomic_write", failing_atomic_write)

    with pytest.raises(OSError, match="injected publication failure"):
        sync_repositories.sync(target, apply=True)

    assert not (target.root / guard).exists()
    assert (target.root / "README.md").read_bytes() == b"stale-unrelated"


def test_partial_publication_group_never_withdraws_guard(tmp_path, monkeypatch):
    _, _, _, guard, _ = _publication_spec("risk-coverage")
    target = _publication_target(tmp_path, (guard,))
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target.root.resolve()
    )
    before = (target.root / guard).read_bytes()

    with pytest.raises(RuntimeError, match="publication group is incomplete"):
        sync_repositories.sync(target, apply=True)

    assert (target.root / guard).read_bytes() == before


def test_three_loss_split_pipeline_is_publicly_allowlisted():
    required_packages = {
        "selectseg/binary_boundary.py",
        "selectseg/binary_diagnostics.py",
        "selectseg/score_binary_common.py",
    }
    required_scripts = {
        "configs/binary_midpoint_dual_pilot.json",
        "scripts/analyze_binary_diagnostics.py",
        "scripts/diagnose_binary_artifact.py",
        "scripts/export_public_provenance.py",
        "scripts/plot_risk_coverage.py",
        "scripts/slurm/diagnose_binary_artifact.sbatch",
        "scripts/slurm/score_binary_common.sbatch",
    }
    required_tests = {
        "tests/test_analyze_binary_diagnostics.py",
        "tests/test_binary_boundary.py",
        "tests/test_binary_diagnostics.py",
        "tests/test_export_public_provenance.py",
        "tests/test_plot_risk_coverage.py",
    }

    assert required_packages <= set(sync_repositories.GITHUB_PACKAGE_FILES)
    assert required_scripts <= set(sync_repositories.GITHUB_SCRIPT_FILES)
    assert required_tests <= set(sync_repositories.GITHUB_TEST_FILES)
    assert "tests/test_sync_repositories.py" not in set(
        sync_repositories.GITHUB_TEST_FILES
    )


def test_schema_v2_main_campaign_is_copied_only_to_github(tmp_path, monkeypatch):
    source = "configs/binary_midpoint_main_v2.json"
    assert source in sync_repositories.GITHUB_SCRIPT_FILES
    assert (sync_repositories.REPO_ROOT / source).is_file()

    declared_targets = sync_repositories._targets()
    github_destinations = {
        item.destination.relative_to(declared_targets["github"].root).as_posix()
        for item in declared_targets["github"].items
    }
    overleaf_destinations = {
        item.destination.relative_to(declared_targets["overleaf"].root).as_posix()
        for item in declared_targets["overleaf"].items
    }
    assert source in github_destinations
    assert source not in overleaf_destinations

    target_root = tmp_path / "github"
    target_root.mkdir()
    target = sync_repositories.SyncTarget(
        "github",
        target_root,
        (
            sync_repositories._item(
                sync_repositories.REPO_ROOT,
                source,
                target_root,
                source,
            ),
        ),
    )
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target_root.resolve()
    )
    assert sync_repositories.sync(target, apply=True) == (1, 0)
    assert (target_root / source).read_bytes() == (
        sync_repositories.REPO_ROOT / source
    ).read_bytes()


def test_scientific_input_execution_and_portable_seals_are_publicly_allowlisted():
    required_packages = {
        "selectseg/scientific_inputs.py",
        "selectseg/binary_artifacts.py",
        "selectseg/data.py",
        "selectseg/freeze_binary_maps.py",
        "selectseg/models.py",
    }
    required_scripts = {
        "configs/binary_midpoint_main_v2.json",
        "configs/binary_midpoint_smoke_v1.json",
        "scripts/submit_binary_simulations.py",
        "scripts/submit_scientific_input_components.py",
        "scripts/slurm/env.sh",
        "scripts/slurm/build_scientific_dataset.sbatch",
        "scripts/slurm/freeze_binary_maps.sbatch",
    }
    required_tests = {
        "tests/test_binary_artifacts.py",
        "tests/test_data.py",
        "tests/test_freeze_binary_maps.py",
        "tests/test_scientific_inputs.py",
        "tests/test_submit_binary_simulations.py",
        "tests/test_submit_scientific_input_components.py",
    }
    expected_seals = {
        "configs/scientific_inputs/binary-midpoint-main-v2/base_models.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/checkpoints.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/datasets/pet.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/datasets/kvasir.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/datasets/fives.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/datasets/isic.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/datasets/tn3k.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/environment.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/source.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json",
        "configs/scientific_inputs/binary-midpoint-smoke-v1/checkpoints.json",
        "configs/scientific_inputs/binary-midpoint-smoke-v1/root.lock.json",
    }

    assert required_packages <= set(sync_repositories.GITHUB_PACKAGE_FILES)
    assert required_scripts <= set(sync_repositories.GITHUB_SCRIPT_FILES)
    assert required_tests <= set(sync_repositories.GITHUB_TEST_FILES)
    assert expected_seals == set(sync_repositories.GITHUB_SCIENTIFIC_INPUT_FILES)
    assert all(path.endswith(".json") for path in expected_seals)
    sync_repositories._validate_public_scientific_input_closure(
        sync_repositories.REPO_ROOT
    )

    targets = sync_repositories._targets()
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    overleaf_destinations = {
        item.destination.relative_to(targets["overleaf"].root).as_posix()
        for item in targets["overleaf"].items
    }
    assert expected_seals | required_packages | required_scripts | required_tests <= (
        github_destinations
    )
    assert not expected_seals & overleaf_destinations


@pytest.mark.parametrize(
    "value",
    (
        "/scratch.global/private/data",
        "C:\\private\\data",
        "https://private.invalid/model",
        "owner=zhan9381",
    ),
)
def test_public_scientific_seal_privacy_guard_rejects_nonportable_values(value):
    with pytest.raises(RuntimeError, match="private or non-portable"):
        sync_repositories._validate_portable_scientific_json(
            _json_bytes({"path": value}), source="injected.json"
        )


def test_preinference_scientific_seals_stay_out_of_analysis_only_artifact():
    release_sources = {
        item.source for item in build_anonymous_analysis_artifact.RELEASE_FILES
    }
    assert not set(sync_repositories.GITHUB_SCIENTIFIC_INPUT_FILES) & release_sources
    assert "selectseg/scientific_inputs.py" not in release_sources


def test_m128_auxiliary_pipeline_and_analysis_are_publicly_allowlisted():
    required_packages = {"selectseg/score_binary_m128_auxiliary.py"}
    required_scripts = {
        "scripts/submit_m128_auxiliary.py",
        "scripts/slurm/score_binary_m128_auxiliary.sbatch",
    }
    required_tests = {"tests/test_score_binary_m128_auxiliary.py"}

    assert required_packages <= set(sync_repositories.GITHUB_PACKAGE_FILES)
    assert required_scripts <= set(sync_repositories.GITHUB_SCRIPT_FILES)
    assert required_tests <= set(sync_repositories.GITHUB_TEST_FILES)
    assert (
        "outputs/binary_m128_auxiliary_analysis/analysis.json",
        "results/m128_numerical_reference.json",
    ) in sync_repositories.GITHUB_RESULT_FILES


def test_seed_and_synthetic_extension_workflows_are_publicly_allowlisted():
    required_packages = {
        "selectseg/binary_seed_downstream.py",
        "selectseg/binary_seed_extension.py",
        "selectseg/synthetic_posterior.py",
    }
    required_scripts = {
        "configs/auxiliary/binary_seed_extension-v1.json",
        "configs/auxiliary/binary_seed_extension-v1.lock.json",
        "configs/auxiliary/synthetic_posterior-v1.json",
        "configs/auxiliary/synthetic_posterior-v1.lock.json",
        "configs/auxiliary/synthetic_posterior-v1.README.md",
        "scripts/submit_binary_seed_extension.py",
        "scripts/collect_binary_seed_diagnostics.py",
        "scripts/analyze_binary_seed_extension.py",
        "scripts/export_binary_seed_provenance.py",
        "scripts/export_seed_replay_bundle.py",
        "scripts/adjust_seed_downstream_timelimits.py",
        "scripts/finalize_seed_scheduler_ledger.py",
        "scripts/render_binary_seed_extension.py",
        "scripts/render_seed_gate_table.py",
        "scripts/replay_seed_robustness.py",
        "scripts/publish_binary_seed_extension.py",
        "scripts/submit_synthetic_posterior.py",
        "scripts/analyze_synthetic_posterior.py",
        "scripts/render_synthetic_posterior.py",
        "scripts/slurm/train_binary_seed_extension.sbatch",
        "scripts/slurm/freeze_binary_seed_extension.sbatch",
        "scripts/slurm/analyze_binary_seed_extension.sbatch",
        "scripts/slurm/render_binary_seed_extension.sbatch",
        "scripts/slurm/run_synthetic_posterior.sbatch",
    }
    required_tests = {
        "tests/test_binary_seed_extension.py",
        "tests/test_collect_binary_seed_diagnostics.py",
        "tests/test_export_binary_seed_provenance.py",
        "tests/test_adjust_seed_downstream_timelimits.py",
        "tests/test_finalize_seed_scheduler_ledger.py",
        "tests/test_publish_binary_seed_extension.py",
        "tests/test_render_seed_gate_table.py",
        "tests/test_replay_seed_robustness.py",
        "tests/test_synthetic_posterior.py",
    }
    assert required_packages <= set(sync_repositories.GITHUB_PACKAGE_FILES)
    assert required_scripts <= set(sync_repositories.GITHUB_SCRIPT_FILES)
    assert required_tests <= set(sync_repositories.GITHUB_TEST_FILES)


def test_anonymous_analysis_artifact_builder_is_publicly_allowlisted_and_sync_safe(
    tmp_path, monkeypatch
):
    source = "scripts/build_anonymous_analysis_artifact.py"
    test = "tests/test_build_anonymous_analysis_artifact.py"
    assert source in sync_repositories.GITHUB_SCRIPT_FILES
    assert test in sync_repositories.GITHUB_TEST_FILES
    assert (sync_repositories.REPO_ROOT / source).is_file()
    assert (sync_repositories.REPO_ROOT / test).is_file()

    target_root = tmp_path / "github"
    target_root.mkdir()
    target = sync_repositories.SyncTarget(
        "github",
        target_root,
        tuple(
            sync_repositories._item(
                sync_repositories.REPO_ROOT,
                path,
                target_root,
                path,
            )
            for path in (source, test)
        ),
    )
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target_root.resolve()
    )
    assert sync_repositories.sync(target, apply=False) == (2, 0)


@pytest.mark.parametrize(
    "payload",
    (
        b"token=" + b"olp" + b"_not-a-real-token\n",
        b"token=" + b"sk" + b"-proj-not-a-real-token\n",
    ),
)
def test_runtime_credential_markers_remain_rejected_by_sync_and_artifact_scans(
    tmp_path, monkeypatch, payload
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "payload.txt").write_bytes(payload)
    target_root = tmp_path / "github"
    target_root.mkdir()
    target = sync_repositories.SyncTarget(
        "github",
        target_root,
        (
            sync_repositories._item(
                source_root,
                "payload.txt",
                target_root,
                "payload.txt",
            ),
        ),
    )
    monkeypatch.setattr(
        sync_repositories, "_verify_git_clone", lambda _: target_root.resolve()
    )
    with pytest.raises(RuntimeError, match="possible credential marker"):
        sync_repositories.sync(target, apply=False)
    with pytest.raises(
        build_anonymous_analysis_artifact.ArtifactValidationError,
        match="credential marker",
    ):
        build_anonymous_analysis_artifact.scan_anonymous_bytes(
            payload, source="payload.txt"
        )


def test_portable_auxiliary_and_reliability_workflows_are_publicly_allowlisted():
    required_scripts = {
        "scripts/export_portable_analysis.py",
        "scripts/analyze_matched_risk_reliability.py",
        "scripts/render_matched_risk_reliability.py",
    }
    required_tests = {
        "tests/test_export_portable_analysis.py",
        "tests/test_matched_risk_reliability.py",
    }
    assert required_scripts <= set(sync_repositories.GITHUB_SCRIPT_FILES)
    assert required_tests <= set(sync_repositories.GITHUB_TEST_FILES)

    result_files = dict(sync_repositories.GITHUB_RESULT_FILES)
    expected_results = {
        "outputs/binary_gamma_sensitivity_analysis/b0d4468443fcc46e/analysis.json": (
            "results/gamma_sensitivity.json"
        ),
        "outputs/binary_cardinality_diagnostics_analysis/analysis.json": (
            "results/cardinality_diagnostics.json"
        ),
        "outputs/binary_matched_risk_reliability/analysis.json": (
            "results/matched_risk_reliability.json"
        ),
        "outputs/public_auxiliary/runtime_v1_analysis.json": (
            "results/runtime_v1_analysis.json"
        ),
        "outputs/public_auxiliary/runtime_v1_export.json": (
            "results/runtime_v1_export.json"
        ),
        "outputs/public_auxiliary/runtime_ladder_v2_analysis.json": (
            "results/runtime_ladder_v2_analysis.json"
        ),
        "outputs/public_auxiliary/runtime_ladder_v2_export.json": (
            "results/runtime_ladder_v2_export.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_analysis.json": (
            "results/synthetic_posterior_analysis.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_export.json": (
            "results/synthetic_posterior_export.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_pilot_analysis.json": (
            "results/synthetic_posterior_pilot_analysis.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_pilot_export.json": (
            "results/synthetic_posterior_pilot_export.json"
        ),
    }
    assert expected_results.items() <= result_files.items()


def test_public_provenance_exporter_and_result_are_github_only():
    source = "scripts/export_public_provenance.py"
    test = "tests/test_export_public_provenance.py"
    assert source in sync_repositories.GITHUB_SCRIPT_FILES
    assert test in sync_repositories.GITHUB_TEST_FILES
    assert (sync_repositories.REPO_ROOT / source).is_file()
    assert (sync_repositories.REPO_ROOT / test).is_file()

    targets = sync_repositories._targets()
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    overleaf_destinations = {
        item.destination.relative_to(targets["overleaf"].root).as_posix()
        for item in targets["overleaf"].items
    }
    assert {source, test} <= github_destinations
    assert not ({source, test} & overleaf_destinations)
    assert (
        "outputs/binary_final_v3_analysis/public_provenance.json",
        "results/public_provenance.json",
    ) in sync_repositories.GITHUB_RESULT_FILES
    assert "results/public_provenance.json" in github_destinations
    assert "results/public_provenance.json" not in overleaf_destinations


def test_final_locked_results_are_exactly_and_portably_allowlisted():
    run_ids = {
        ("fives", "clipseg-general"): "74db5fc8c0672236",
        ("fives", "clipseg-target"): "627922c22aa3aa21",
        ("fives", "deeplabv3-target"): "a62d754378787146",
        ("isic", "clipseg-general"): "500bf2dd2f3564b2",
        ("isic", "clipseg-target"): "69fc681ff737f148",
        ("isic", "deeplabv3-target"): "a42288ea78a340b1",
        ("kvasir", "clipseg-general"): "8f5786edfc8d5993",
        ("kvasir", "clipseg-target"): "1a1c02b968ab86a4",
        ("kvasir", "deeplabv3-target"): "5cca940975b7f02d",
        ("pet", "clipseg-general"): "88d2033138fc971a",
        ("pet", "clipseg-target"): "cd4341aaed2bda20",
        ("pet", "deeplabv3-external"): "014bfe3ad787f51e",
        ("pet", "deeplabv3-target"): "fd2f61c609c18fba",
        ("tn3k", "clipseg-general"): "4faed2465c790f8c",
        ("tn3k", "clipseg-target"): "eccb8f4f045a5473",
        ("tn3k", "deeplabv3-target"): "01b4c3a58986cf27",
    }
    frozen_ids = {
        ("pet", "clipseg-general"): "e0d454ce916c5ca9",
        ("pet", "clipseg-target"): "7261edbb5f763435",
        ("pet", "deeplabv3-target"): "ced4dc6e67bc178f",
        ("pet", "deeplabv3-external"): "640ef98bcd028dc3",
        ("kvasir", "clipseg-general"): "eb7304bdd945035d",
        ("kvasir", "clipseg-target"): "7ce9cddd0f9704fd",
        ("kvasir", "deeplabv3-target"): "be7dedef6990d54c",
        ("fives", "clipseg-general"): "cb5c08e73819574c",
        ("fives", "clipseg-target"): "ef8e1b5176b92b87",
        ("fives", "deeplabv3-target"): "e7e1e90e13ce3dfa",
        ("isic", "clipseg-general"): "3e57aa259264c1be",
        ("isic", "clipseg-target"): "a5d30aa0ec8b0742",
        ("isic", "deeplabv3-target"): "48cf75646c61ecf4",
        ("tn3k", "clipseg-general"): "9626e945ef94b1e5",
        ("tn3k", "clipseg-target"): "603dc94883b5f395",
        ("tn3k", "deeplabv3-target"): "efce266864cd5269",
    }
    expected = {
        "outputs/binary_campaign/campaign.lock.json": "results/campaign.lock.json",
        "outputs/binary_final_v3_analysis/analysis.json": "results/analysis.json",
        "outputs/binary_final_v3_analysis/main_table.csv": "results/main_table.csv",
        "outputs/binary_final_v3_analysis/public_provenance.json": (
            "results/public_provenance.json"
        ),
        "outputs/binary_final_v2_analysis/analysis.json": "results/analysis_v2.json",
        "outputs/binary_final_v2_analysis/public_provenance.json": (
            "results/public_provenance_v2.json"
        ),
        "outputs/binary_final_v2_diagnostics/diagnostics_analysis.json": (
            "results/diagnostics_analysis.json"
        ),
        "outputs/binary_working_risk_diagnostics_v2/diagnostics.json": (
            "results/working_risk_diagnostics.json"
        ),
        "outputs/binary_m128_auxiliary_analysis/analysis.json": (
            "results/m128_numerical_reference.json"
        ),
        "outputs/binary_qualitative_cases/f618c91bfeaa467e/selection.json": (
            "results/qualitative_selection.json"
        ),
        "outputs/binary_gamma_sensitivity_analysis/b0d4468443fcc46e/analysis.json": (
            "results/gamma_sensitivity.json"
        ),
        "outputs/binary_cardinality_diagnostics_analysis/analysis.json": (
            "results/cardinality_diagnostics.json"
        ),
        "outputs/binary_matched_risk_reliability/analysis.json": (
            "results/matched_risk_reliability.json"
        ),
        "outputs/public_auxiliary/runtime_v1_analysis.json": (
            "results/runtime_v1_analysis.json"
        ),
        "outputs/public_auxiliary/runtime_v1_export.json": (
            "results/runtime_v1_export.json"
        ),
        "outputs/public_auxiliary/runtime_ladder_v2_analysis.json": (
            "results/runtime_ladder_v2_analysis.json"
        ),
        "outputs/public_auxiliary/runtime_ladder_v2_export.json": (
            "results/runtime_ladder_v2_export.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_analysis.json": (
            "results/synthetic_posterior_analysis.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_export.json": (
            "results/synthetic_posterior_export.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_pilot_analysis.json": (
            "results/synthetic_posterior_pilot_analysis.json"
        ),
        "outputs/public_auxiliary/synthetic_posterior_pilot_export.json": (
            "results/synthetic_posterior_pilot_export.json"
        ),
    }
    for dataset in ("fives", "isic", "kvasir", "pet", "tn3k"):
        for model in ("clipseg", "deeplabv3"):
            relative = f"{dataset}/{model}/seed-0/train_config.json"
            expected[f"outputs/binary_train/{relative}"] = (
                f"results/training/{relative}"
            )
    for dataset in ("isic", "tn3k"):
        for model in ("clipseg", "deeplabv3"):
            relative = f"{dataset}/{model}/seed-0/history.json"
            expected[f"outputs/binary_train/{relative}"] = (
                f"results/training/{relative}"
            )
    for (dataset, condition), artifact_id in frozen_ids.items():
        relative = f"{dataset}/{condition}/{artifact_id}/manifest.json"
        path = f"outputs/binary_artifacts/{relative}"
        expected[path] = path
    for (dataset, condition), run_id in run_ids.items():
        for name in ("manifest.json", "records.jsonl"):
            relative = f"{dataset}/{condition}/{run_id}/{name}"
            expected[f"outputs/binary_assembled/{relative}"] = (
                f"results/assembled/{relative}"
            )

    compatibility_pairs = {
        (
            "outputs/binary_campaign/campaign.lock.json",
            "outputs/binary_campaign/campaign.lock.json",
        )
    }
    compatibility_pairs.update(
        (source, source)
        for source in expected
        if source.startswith("outputs/binary_train/")
        and source.endswith("/train_config.json")
    )
    expected_pairs = set(expected.items()) | compatibility_pairs
    result_pairs = set(sync_repositories.GITHUB_RESULT_FILES)
    assert len(sync_repositories.GITHUB_RESULT_FILES) == 94
    assert result_pairs == expected_pairs
    assert all(
        (sync_repositories.REPO_ROOT / source).is_file() for source, _ in result_pairs
    )
    assert not any(
        "receipt" in source or "receipt" in target for source, target in result_pairs
    )

    targets = sync_repositories._targets()
    github_destinations = {
        item.destination.relative_to(targets["github"].root).as_posix()
        for item in targets["github"].items
    }
    assert "outputs/binary_campaign/campaign.lock.json" in github_destinations


def test_guarded_diagnostics_table_has_an_explicit_optional_allowlist():
    assert sync_repositories.OPTIONAL_GENERATED_MANUSCRIPT_FILES == (
        "Tables/binary_diagnostics.tex",
        "Tables/cardinality_diagnostics.tex",
        "Tables/gamma_sensitivity.tex",
        "Tables/working_risk_diagnostics.tex",
        "Tables/m128_numerical_reference.tex",
        "Tables/binary_runtime.tex",
        "Tables/seed_robustness.tex",
        "Tables/seed_sensitivity_main.tex",
        "Tables/synthetic_posterior_summary.tex",
        "Figures/synthetic_posterior_summary.pdf",
        "Figures/matched_risk_reliability_pet.pdf",
        "Figures/matched_risk_reliability_kvasir.pdf",
        "Figures/matched_risk_reliability_fives.pdf",
        "Figures/matched_risk_reliability_isic.pdf",
        "Figures/matched_risk_reliability_tn3k.pdf",
    )
    assert not (
        set(sync_repositories.OPTIONAL_GENERATED_MANUSCRIPT_FILES)
        & set(sync_repositories.MANUSCRIPT_FILES)
    )


def test_stale_unreachable_auxiliary_table_is_not_mirrored():
    assert "Tables/auxiliary_experiments.tex" not in set(
        sync_repositories.MANUSCRIPT_FILES
    )


def test_only_the_completed_pilot_gate_is_declared_as_a_canonical_result():
    sources = {source for source, _ in sync_repositories.GITHUB_RESULT_FILES}
    pilot_sources = {source for source in sources if "pilot" in source}
    assert pilot_sources == {
        "outputs/public_auxiliary/synthetic_posterior_pilot_analysis.json",
        "outputs/public_auxiliary/synthetic_posterior_pilot_export.json",
    }
    assert not any(
        marker in source
        for source in sources
        for marker in ("preliminary", "smoke", "binary_final/analysis")
    )


def test_allowlists_do_not_contain_duplicates():
    for paths in (
        sync_repositories.MANUSCRIPT_FILES,
        sync_repositories.MANUSCRIPT_PDF_FILES,
        sync_repositories.OPTIONAL_GENERATED_MANUSCRIPT_FILES,
        sync_repositories.OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES,
        sync_repositories.OPTIONAL_GENERATED_FIGURE_SENTINEL_FILES,
        sync_repositories.OPTIONAL_GENERATED_FIGURE_MANIFEST_FILES,
        sync_repositories.OPTIONAL_GENERATED_QUALITATIVE_FILES,
        sync_repositories.GITHUB_ROOT_FILES,
        sync_repositories.GITHUB_PACKAGE_FILES,
        sync_repositories.GITHUB_SCRIPT_FILES,
        sync_repositories.GITHUB_SCIENTIFIC_INPUT_FILES,
        sync_repositories.GITHUB_TEST_FILES,
    ):
        assert len(paths) == len(set(paths))
    public_seed_sources = [
        source for source, _ in sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES
    ]
    public_seed_targets = [
        target for _, target in sync_repositories.OPTIONAL_PUBLIC_SEED_RESULT_FILES
    ]
    assert len(public_seed_sources) == len(set(public_seed_sources))
    assert len(public_seed_targets) == len(set(public_seed_targets))
    replay_sources = [
        source
        for source, _ in sync_repositories.OPTIONAL_PUBLIC_SEED_REPLAY_FILES
    ]
    replay_targets = [
        target
        for _, target in sync_repositories.OPTIONAL_PUBLIC_SEED_REPLAY_FILES
    ]
    assert len(replay_sources) == len(set(replay_sources)) == 62
    assert len(replay_targets) == len(set(replay_targets)) == 62
    result_sources = [source for source, _ in sync_repositories.GITHUB_RESULT_FILES]
    result_targets = [target for _, target in sync_repositories.GITHUB_RESULT_FILES]
    duplicate_sources = {
        source for source in result_sources if result_sources.count(source) > 1
    }
    expected_duplicates = {"outputs/binary_campaign/campaign.lock.json"}
    expected_duplicates.update(
        f"outputs/binary_train/{dataset}/{model}/seed-0/train_config.json"
        for dataset in ("fives", "isic", "kvasir", "pet", "tn3k")
        for model in ("clipseg", "deeplabv3")
    )
    assert duplicate_sources == expected_duplicates
    assert all(result_sources.count(source) == 2 for source in duplicate_sources)
    assert len(result_targets) == len(set(result_targets))
