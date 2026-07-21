"""Repository-relative path handling for tracked experiment specifications.

Early auxiliary specifications were executed from an ignored nested checkout
and therefore contain paths beginning with ``../``.  The repository root is
now the only code checkout.  This module preserves the immutable specification
bytes while resolving both legacy and canonical relative paths consistently.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repository_path(value: str | Path) -> Path:
    """Resolve a specification path against the canonical repository root."""

    path = Path(value)
    if path.is_absolute():
        return path
    parts = list(path.parts)
    while parts and parts[0] in {".", ".."}:
        parts.pop(0)
    return REPOSITORY_ROOT.joinpath(*parts)
