#!/usr/bin/env python3
"""CLI wrapper for one frozen binary-artifact diagnostic job."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selectseg.binary_diagnostics import main  # noqa: E402


if __name__ == "__main__":
    main()
