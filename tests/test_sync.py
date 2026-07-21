import pytest

from scripts import sync


def test_discovery_excludes_latex_build_products(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    (docs / "Sections").mkdir(parents=True)
    (docs / "Sections/main.tex").write_text("source")
    (docs / "Sections/main.aux").write_text("build")
    monkeypatch.setattr(sync, "DOCS", docs)
    assert [path.name for path in sync.manuscript_files()] == ["main.tex"]


def test_checkout_guard_rejects_missing_git_directory(tmp_path):
    with pytest.raises(RuntimeError):
        sync._verify_checkout(tmp_path)


def test_pending_changes_preserves_unmanaged_files(monkeypatch, tmp_path):
    docs = tmp_path / "docs"
    overleaf = tmp_path / "overleaf"
    (docs / "Sections").mkdir(parents=True)
    (overleaf / ".git").mkdir(parents=True)
    (overleaf / "unmanaged.txt").write_text("keep")
    (docs / "Sections/main.tex").write_text("new")
    monkeypatch.setattr(sync, "DOCS", docs)
    monkeypatch.setattr(sync, "OVERLEAF", overleaf)
    changes = sync.pending_changes()
    assert len(changes) == 1
    sync._copy_atomic(changes[0])
    assert (overleaf / "Sections/main.tex").read_text() == "new"
    assert (overleaf / "unmanaged.txt").read_text() == "keep"
