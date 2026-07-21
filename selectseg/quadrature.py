"""Strict threshold-estimator specifications for binary simulations.

The distribution defining the working posterior and the numerical rule used
to integrate against it are deliberately kept separate.  ``midpoint-v1``
targets the uniform-threshold integral with a deterministic midpoint rule.  A
future non-uniform proposal must therefore be introduced as a new, explicitly
importance-weighted estimator rather than silently changing this target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ESTIMATOR_SPEC_SCHEMA_VERSION = 1
_SPEC_KEYS = {
    "schema_version",
    "estimator_id",
    "target_measure",
    "rule",
    "randomized",
    "required_seed",
}


@dataclass(frozen=True)
class EstimatorSpec:
    """Validated estimator metadata loaded from one immutable JSON spec."""

    schema_version: int
    estimator_id: str
    target_measure: str
    rule: str
    randomized: bool
    required_seed: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class ThresholdRule:
    """The nodes and normalized weights for exactly one simulation."""

    estimator_id: str
    m: int
    seed: int
    nodes: np.ndarray
    weights: np.ndarray


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without normalizing its bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _parse_spec(payload: Mapping[str, Any], *, path: Path, sha256: str) -> EstimatorSpec:
    if not isinstance(payload, Mapping):
        raise TypeError("estimator spec must be a JSON object")
    keys = set(payload)
    missing = _SPEC_KEYS - keys
    extra = keys - _SPEC_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"unknown={sorted(extra)}")
        raise ValueError("invalid estimator spec keys: " + ", ".join(details))

    schema_version = _strict_int(
        payload["schema_version"], name="schema_version", minimum=1
    )
    if schema_version != ESTIMATOR_SPEC_SCHEMA_VERSION:
        raise ValueError(
            "unsupported estimator spec schema_version: "
            f"{schema_version} != {ESTIMATOR_SPEC_SCHEMA_VERSION}"
        )
    for key in ("estimator_id", "target_measure", "rule"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise TypeError(f"{key} must be a non-empty string")
    if not isinstance(payload["randomized"], bool):
        raise TypeError("randomized must be a boolean")
    required_seed = _strict_int(payload["required_seed"], name="required_seed")

    spec = EstimatorSpec(
        schema_version=schema_version,
        estimator_id=payload["estimator_id"],
        target_measure=payload["target_measure"],
        rule=payload["rule"],
        randomized=payload["randomized"],
        required_seed=required_seed,
        path=path,
        sha256=sha256,
    )
    if spec.estimator_id != "midpoint-v1":
        raise ValueError(f"unsupported estimator_id: {spec.estimator_id!r}")
    if (
        spec.target_measure != "uniform-threshold"
        or spec.rule != "midpoint"
        or spec.randomized
        or spec.required_seed != 0
    ):
        raise ValueError(
            "midpoint-v1 must be deterministic midpoint quadrature for the "
            "uniform-threshold target with required_seed=0"
        )
    return spec


def load_estimator_spec(path: str | Path) -> EstimatorSpec:
    """Load and strictly validate one estimator specification."""

    spec_path = Path(path).resolve()
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid estimator JSON: {spec_path}") from error
    return _parse_spec(payload, path=spec_path, sha256=sha256_file(spec_path))


def build_threshold_rule(spec: EstimatorSpec, *, m: int, seed: int) -> ThresholdRule:
    """Construct the one rule authorized by ``spec`` for ``m`` and ``seed``."""

    count = _strict_int(m, name="m", minimum=1)
    simulation_seed = _strict_int(seed, name="seed")
    if simulation_seed != spec.required_seed:
        raise ValueError(
            f"{spec.estimator_id} requires seed={spec.required_seed}; "
            f"received seed={simulation_seed}"
        )
    if spec.estimator_id != "midpoint-v1":
        raise ValueError(f"unsupported estimator_id: {spec.estimator_id!r}")

    nodes = (np.arange(count, dtype=float) + 0.5) / count
    weights = np.full(count, 1.0 / count, dtype=float)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return ThresholdRule(
        estimator_id=spec.estimator_id,
        m=count,
        seed=simulation_seed,
        nodes=nodes,
        weights=weights,
    )


__all__ = [
    "ESTIMATOR_SPEC_SCHEMA_VERSION",
    "EstimatorSpec",
    "ThresholdRule",
    "build_threshold_rule",
    "load_estimator_spec",
    "sha256_file",
]
