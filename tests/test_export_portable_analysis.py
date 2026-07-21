import hashlib
import json

import pytest

from scripts.maintenance.export_analysis import export_analysis


def _root(tmp_path):
    (tmp_path / "README.md").write_text("root\n", encoding="utf-8")
    return tmp_path


def _write_source(root, payload):
    source = root / "private" / "analysis.json"
    source.parent.mkdir()
    source.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_export_rewrites_only_repository_absolute_paths_and_binds_hashes(tmp_path):
    root = _root(tmp_path)
    absolute = root / "outputs" / "run" / "records.jsonl"
    source, digest = _write_source(
        root,
        {
            "path": absolute.as_posix(),
            "relative": "outputs/already-portable.json",
            "values": [0.1, 0.2],
        },
    )
    output = root / "public" / "analysis.json"
    manifest = root / "public" / "analysis.export.json"
    _, _, record = export_analysis(
        source,
        expected_source_sha256=digest,
        output=output,
        manifest=manifest,
        repository_root=root,
    )

    portable = json.loads(output.read_text())
    assert portable["path"] == "outputs/run/records.jsonl"
    assert portable["relative"] == "outputs/already-portable.json"
    assert record["rewritten_absolute_path_count"] == 1
    assert json.loads(manifest.read_text()) == record


@pytest.mark.parametrize("field", ["expected", "outside", "secret"])
def test_export_rejects_invalid_source_before_writing(tmp_path, field):
    root = _root(tmp_path)
    value = {"ok": True}
    if field == "outside":
        value["path"] = "/etc/passwd"
    if field == "secret":
        value["credential"] = "ghp" + "_example"
    source, digest = _write_source(root, value)
    expected = "0" * 64 if field == "expected" else digest
    output = root / "public" / "analysis.json"
    manifest = root / "public" / "analysis.export.json"

    with pytest.raises(ValueError):
        export_analysis(
            source,
            expected_source_sha256=expected,
            output=output,
            manifest=manifest,
            repository_root=root,
        )
    assert not output.exists()
    assert not manifest.exists()


def test_export_rejects_duplicate_keys_nonfinite_and_overwrite(tmp_path):
    root = _root(tmp_path)
    source = root / "analysis.json"
    output = root / "public.json"
    manifest = root / "manifest.json"
    source.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        export_analysis(
            source,
            expected_source_sha256=digest,
            output=output,
            manifest=manifest,
            repository_root=root,
        )

    source.write_text('{"x":NaN}\n', encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="non-standard"):
        export_analysis(
            source,
            expected_source_sha256=digest,
            output=output,
            manifest=manifest,
            repository_root=root,
        )

    source, digest = _write_source(root, {"x": 1})
    output.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_analysis(
            source,
            expected_source_sha256=digest,
            output=output,
            manifest=manifest,
            repository_root=root,
        )


def test_export_rejects_symlink_inputs_and_destinations(tmp_path):
    root = _root(tmp_path)
    source, digest = _write_source(root, {"x": 1})
    source_link = root / "source-link.json"
    source_link.symlink_to(source)
    with pytest.raises(FileNotFoundError):
        export_analysis(
            source_link,
            expected_source_sha256=digest,
            output=root / "out.json",
            manifest=root / "manifest.json",
            repository_root=root,
        )

    linked_parent = root / "linked"
    real_parent = root / "real"
    real_parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_analysis(
            source,
            expected_source_sha256=digest,
            output=linked_parent / "out.json",
            manifest=root / "manifest.json",
            repository_root=root,
        )
