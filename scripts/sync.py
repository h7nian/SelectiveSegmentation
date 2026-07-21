"""Preview or copy the public manuscript mirror into the Overleaf checkout."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OVERLEAF = ROOT / "overleaf"
ROOT_FILES = (
    "README.md",
    "main.tex",
    "math_commands.tex",
    "references.bib",
    "fancyhdr.sty",
    "iclr2026_conference.bst",
    "iclr2026_conference.sty",
    "natbib.sty",
)
TREE_RULES = {
    "Sections": {".tex"},
    "Tables": {".tex"},
    "Figures": {".tex", ".json", ".pdf", ".png", ".jpg", ".jpeg"},
}


@dataclass(frozen=True)
class Change:
    source: Path
    destination: Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _verify_checkout(path: Path) -> None:
    if path.is_symlink() or not (path / ".git").is_dir():
        raise RuntimeError(f"{path} is not a regular Git checkout")


def manuscript_files() -> tuple[Path, ...]:
    files = [DOCS / name for name in ROOT_FILES if (DOCS / name).is_file()]
    for directory, suffixes in TREE_RULES.items():
        root = DOCS / directory
        files.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in suffixes
        )
    relative = [path.relative_to(DOCS) for path in files]
    if len(relative) != len(set(relative)):
        raise RuntimeError("manuscript file discovery produced duplicates")
    return tuple(files)


def pending_changes() -> tuple[Change, ...]:
    _verify_checkout(OVERLEAF)
    changes = []
    for source in manuscript_files():
        destination = OVERLEAF / source.relative_to(DOCS)
        if destination.is_symlink():
            raise RuntimeError(f"refusing to replace symlink: {destination}")
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            changes.append(Change(source, destination))
    return tuple(changes)


def _copy_atomic(change: Change) -> None:
    change.destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{change.destination.name}.", dir=change.destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(change.source.read_bytes())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, change.destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    args = parse_args(argv)
    changes = pending_changes()
    for change in changes:
        print(change.destination.relative_to(OVERLEAF))
        if args.apply:
            _copy_atomic(change)
    print(f"{len(changes)} change(s); apply={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
