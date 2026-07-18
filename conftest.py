"""Root pytest configuration.

Living at the repository root keeps ``selectseg`` importable in tests, and
pins the model caches to the repo so tests never download from the network
when the caches are already populated.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

os.environ["HF_HOME"] = str(REPO_ROOT / "data" / "cache" / "huggingface")
os.environ["TORCH_HOME"] = str(REPO_ROOT / "data" / "cache" / "torch")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
