"""Known-posterior stress test for the shared-threshold working posterior.

One invocation evaluates exactly one frozen grid cell.  It constructs a small
cohort of probability maps, keeps the deployed action fixed at ``p >= gamma``,
and compares the working shared-threshold posterior ``Q_p`` with one of four
true mask couplings having the same pixel marginals.  Posterior masks are never
written: the artifact contains only aggregate sufficient statistics and a
strict manifest.

The scalar loss-pushforward Wasserstein diagnostic is valid for Dice, nHD, and
nHD95.  The mask-level paired transport diagnostics are explicitly upper
bounds for Jaccard and normalized full-HD costs; no HD95 Wasserstein corollary
is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import string
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

from selectseg.geometry import normalized_penalized_boundary_losses
from selectseg.confidence import tie_aware_expected_aurc
from selectseg.quadrature import sha256_file


COUPLINGS = (
    "shared_threshold",
    "independent_bernoulli",
    "local_block_threshold",
    "bimodal_antithetic",
)
SHARPNESS_LEVELS = ("diffuse", "medium", "sharp")
MORPHOLOGIES = ("disk", "elongated", "two_component")
M_VALUES = (2, 8, 32, 128)
LOSSES = ("dice", "nhd", "nhd95")
CPU_PARTITIONS = ("agsmall", "amdsmall", "msismall")
SUMMARY_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.synthetic_cell"

_SPEC_FIELDS = frozenset(
    {
        "spec_schema_version",
        "campaign_id",
        "base_seed",
        "grid",
        "pilot",
        "protocol",
        "cpu_partitions",
        "paths",
    }
)
_LOCK_FIELDS = frozenset({"lock_schema_version", "campaign_id", "spec", "code_sources"})


@dataclass(frozen=True, slots=True)
class Cell:
    coupling: str
    sharpness: str
    morphology: str
    replicate: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.coupling, self.sharpness, self.morphology, self.replicate

    @property
    def slug(self) -> str:
        return (
            f"{self.coupling}--{self.sharpness}--{self.morphology}--"
            f"r{self.replicate:02d}"
        )


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path, *, name):
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular file: {source}")
    try:
        data = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {name} {source}: {error}") from error
    if not isinstance(data, dict):
        raise TypeError(f"{name} must contain one object")
    return source, data


def _strict_sha256(value, *, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    if any(character not in string.hexdigits for character in value):
        raise ValueError(f"{name} must be hexadecimal")
    return value.lower()


def _resolve_from_repo(lock_path, value):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    cwd = (Path.cwd() / path).resolve()
    repository = (Path(lock_path).resolve().parents[2] / path).resolve()
    return cwd if cwd.exists() or not repository.exists() else repository


def _validate_spec(spec):
    if set(spec) != _SPEC_FIELDS or spec.get("spec_schema_version") != 1:
        raise ValueError("synthetic spec has an unexpected schema")
    if spec.get("campaign_id") != "binary-synthetic-coupling-v1":
        raise ValueError("unexpected synthetic campaign_id")
    if isinstance(spec.get("base_seed"), bool) or not isinstance(
        spec.get("base_seed"), int
    ):
        raise TypeError("base_seed must be an integer")
    expected_grid = {
        "couplings": list(COUPLINGS),
        "sharpness_levels": list(SHARPNESS_LEVELS),
        "morphologies": list(MORPHOLOGIES),
        "replicates": 10,
    }
    if spec.get("grid") != expected_grid:
        raise ValueError("synthetic grid differs from the predeclared 360 cells")
    if spec.get("pilot") != {"morphology": "disk", "replicate": 0}:
        raise ValueError("pilot must be the predeclared 12-cell subset")
    protocol = spec.get("protocol")
    required_protocol = {
        "height",
        "width",
        "cohort_size",
        "posterior_draws",
        "mc_batches",
        "workers_per_job",
        "gamma",
        "m_values",
        "sharpness_pixels",
        "local_block_size",
        "losses",
        "empty_convention",
        "distance_normalization",
        "threshold_rule",
        "dice_exact",
    }
    if not isinstance(protocol, dict) or set(protocol) != required_protocol:
        raise ValueError("synthetic protocol has unexpected fields")
    for name in (
        "height",
        "width",
        "cohort_size",
        "posterior_draws",
        "mc_batches",
        "workers_per_job",
        "local_block_size",
    ):
        if isinstance(protocol[name], bool) or not isinstance(protocol[name], int):
            raise TypeError(f"protocol.{name} must be an integer")
        if protocol[name] < 1:
            raise ValueError(f"protocol.{name} must be positive")
    if protocol["posterior_draws"] % protocol["mc_batches"]:
        raise ValueError("posterior_draws must be divisible by mc_batches")
    if protocol["m_values"] != list(M_VALUES) or protocol["losses"] != list(LOSSES):
        raise ValueError("frozen M values or losses changed")
    if protocol["gamma"] != 0.5 or protocol["dice_exact"] is not True:
        raise ValueError("frozen action or exact-Dice protocol changed")
    if set(protocol["sharpness_pixels"]) != set(SHARPNESS_LEVELS):
        raise ValueError("sharpness map and grid disagree")
    if tuple(spec.get("cpu_partitions", ())) != CPU_PARTITIONS:
        raise ValueError("unexpected CPU partitions")
    if not isinstance(spec.get("paths"), dict) or set(spec["paths"]) != {
        "pilot_output_root",
        "main_output_root",
        "analysis_output_root",
    }:
        raise ValueError("synthetic output paths have an unexpected schema")


def load_synthetic_lock(path, *, expected_sha256=None):
    """Load the immutable lock, spec, and bound source hashes strictly."""

    lock_path, lock = _load_json(path, name="synthetic lock")
    if set(lock) != _LOCK_FIELDS or lock.get("lock_schema_version") != 1:
        raise ValueError("synthetic lock has an unexpected schema")
    lock_sha = sha256_file(lock_path)
    if expected_sha256 is not None and lock_sha != _strict_sha256(
        expected_sha256, name="expected lock SHA-256"
    ):
        raise ValueError("synthetic lock SHA-256 mismatch")
    if lock.get("campaign_id") != "binary-synthetic-coupling-v1":
        raise ValueError("synthetic lock campaign_id changed")
    binding = lock.get("spec")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("synthetic lock spec binding is invalid")
    spec_path = _resolve_from_repo(lock_path, binding["path"])
    if sha256_file(spec_path) != _strict_sha256(binding["sha256"], name="spec hash"):
        raise ValueError("synthetic spec bytes changed")
    _, spec = _load_json(spec_path, name="synthetic spec")
    _validate_spec(spec)
    if spec["campaign_id"] != lock["campaign_id"]:
        raise ValueError("synthetic spec and lock campaign IDs disagree")
    code = lock.get("code_sources")
    if not isinstance(code, list) or not code:
        raise ValueError("synthetic lock must bind source files")
    seen = set()
    resolved_code = []
    for index, source in enumerate(code):
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise ValueError(f"code_sources[{index}] has an invalid schema")
        source_path = _resolve_from_repo(lock_path, source["path"])
        expected = _strict_sha256(source["sha256"], name=f"code_sources[{index}]")
        if str(source_path) in seen:
            raise ValueError("duplicate source path in synthetic lock")
        seen.add(str(source_path))
        if sha256_file(source_path) != expected:
            raise ValueError(f"bound source bytes changed: {source_path}")
        resolved_code.append((source_path, expected))
    return {
        "path": lock_path,
        "sha256": lock_sha,
        "lock": lock,
        "spec_path": spec_path,
        "spec_sha256": binding["sha256"],
        "spec": spec,
        "code_sources": tuple(resolved_code),
    }


def all_cells(spec) -> tuple[Cell, ...]:
    return tuple(
        Cell(coupling, sharpness, morphology, replicate)
        for coupling in spec["grid"]["couplings"]
        for sharpness in spec["grid"]["sharpness_levels"]
        for morphology in spec["grid"]["morphologies"]
        for replicate in range(spec["grid"]["replicates"])
    )


def pilot_cells(spec) -> tuple[Cell, ...]:
    pilot = spec["pilot"]
    return tuple(
        Cell(coupling, sharpness, pilot["morphology"], pilot["replicate"])
        for coupling in spec["grid"]["couplings"]
        for sharpness in spec["grid"]["sharpness_levels"]
    )


def selected_cells(spec, phase) -> tuple[Cell, ...]:
    if phase == "pilot":
        return pilot_cells(spec)
    if phase == "full":
        pilot = set(pilot_cells(spec))
        return tuple(cell for cell in all_cells(spec) if cell not in pilot)
    if phase == "complete":
        return all_cells(spec)
    raise ValueError("phase must be pilot, full, or complete")


def _seed_from_parts(base_seed, *parts):
    payload = json.dumps([base_seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def cell_seeds(spec, cell):
    common = (cell.sharpness, cell.morphology, cell.replicate)
    return {
        "cell_seed": _seed_from_parts(spec["base_seed"], *cell.key),
        "map_seed": _seed_from_parts(spec["base_seed"], "map", *common),
        "posterior_seed": _seed_from_parts(spec["base_seed"], "posterior", *common),
    }


def validate_cell(spec, cell):
    if cell not in set(all_cells(spec)):
        raise ValueError(f"cell is outside the frozen grid: {cell.key}")


def _sigmoid(value):
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def generate_probability_map(protocol, morphology, sharpness, rng, image_index):
    """Generate one controlled probability map with a nonempty fixed action."""

    height, width = protocol["height"], protocol["width"]
    yy, xx = np.mgrid[:height, :width]
    cx = (width - 1) / 2 + rng.uniform(-3.0, 3.0)
    cy = (height - 1) / 2 + rng.uniform(-3.0, 3.0)
    difficulty = math.exp(rng.uniform(math.log(0.7), math.log(1.45)))
    if morphology == "disk":
        radius = rng.uniform(5.0, 10.0)
        signed = radius - np.hypot(xx - cx, yy - cy)
    elif morphology == "elongated":
        angle = rng.uniform(-0.8, 0.8)
        cosine, sine = math.cos(angle), math.sin(angle)
        xr = cosine * (xx - cx) + sine * (yy - cy)
        yr = -sine * (xx - cx) + cosine * (yy - cy)
        major, minor = rng.uniform(9.0, 13.0), rng.uniform(2.5, 5.0)
        radial = np.sqrt((xr / major) ** 2 + (yr / minor) ** 2)
        signed = (1.0 - radial) * minor
    elif morphology == "two_component":
        separation = rng.uniform(7.0, 12.0)
        angle = rng.uniform(0.0, math.pi)
        dx, dy = math.cos(angle) * separation / 2, math.sin(angle) * separation / 2
        radius_a, radius_b = rng.uniform(3.5, 6.0, size=2)
        signed_a = radius_a - np.hypot(xx - (cx - dx), yy - (cy - dy))
        signed_b = radius_b - np.hypot(xx - (cx + dx), yy - (cy + dy))
        signed = np.maximum(signed_a, signed_b)
    else:
        raise ValueError(f"unknown morphology {morphology!r}")
    phase = rng.uniform(0.0, 2 * math.pi)
    low_frequency = (
        0.35 * np.sin((xx + image_index) / 5.0 + phase) * np.cos(yy / 6.0 - phase)
    )
    temperature = protocol["sharpness_pixels"][sharpness] * difficulty
    probability = _sigmoid((signed + low_frequency) / temperature)
    probability = np.clip(probability, 1e-6, 1 - 1e-6).astype(np.float64)
    if not np.any(probability >= protocol["gamma"]):
        raise RuntimeError("synthetic generator unexpectedly produced an empty action")
    return probability


def _bimodal_groups(shape):
    _, width = shape
    return np.broadcast_to(np.arange(width) < width / 2, shape)


def sample_p_q(probability, coupling, global_uniforms, auxiliary_rng, block_size):
    """Return paired P and Q draws; both posteriors have marginals ``p``."""

    draws = global_uniforms.size
    q_masks = probability[None, :, :] >= global_uniforms[:, None, None]
    if coupling == "shared_threshold":
        return q_masks.copy(), q_masks
    if coupling == "independent_bernoulli":
        p_masks = auxiliary_rng.random((draws, *probability.shape)) <= probability
        return p_masks, q_masks
    if coupling == "local_block_threshold":
        height, width = probability.shape
        block_rows = math.ceil(height / block_size)
        block_cols = math.ceil(width / block_size)
        # The modulo shift retains independent Uniform(0,1) block thresholds
        # while coupling them reproducibly to the global Q draw.
        offsets = auxiliary_rng.random((draws, block_rows, block_cols))
        block_u = (global_uniforms[:, None, None] + offsets) % 1.0
        expanded = np.repeat(
            np.repeat(block_u, block_size, axis=1), block_size, axis=2
        )[:, :height, :width]
        return probability[None, :, :] >= expanded, q_masks
    if coupling == "bimodal_antithetic":
        left = _bimodal_groups(probability.shape)
        thresholds = np.where(
            left[None, :, :],
            global_uniforms[:, None, None],
            1.0 - global_uniforms[:, None, None],
        )
        return probability[None, :, :] >= thresholds, q_masks
    raise ValueError(f"unknown coupling {coupling!r}")


def dice_loss(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = int(first.sum()) + int(second.sum())
    if denominator == 0:
        return 0.0
    return float(1.0 - 2.0 * np.logical_and(first, second).sum() / denominator)


def jaccard_distance(first, second):
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 0.0
    return float(1.0 - np.logical_and(first, second).sum() / union)


def exact_dice_q_risk(probability, action):
    """Exact Uniform-threshold expectation of Dice loss by probability knots."""

    flat_p = np.asarray(probability, dtype=float).ravel()
    flat_a = np.asarray(action, dtype=bool).ravel()
    order = np.argsort(-flat_p, kind="stable")
    sorted_p = flat_p[order]
    sorted_a = flat_a[order]
    action_size = int(flat_a.sum())
    risk = (1.0 - float(sorted_p[0])) * (0.0 if action_size == 0 else 1.0)
    count = 0
    overlap = 0
    start = 0
    while start < sorted_p.size:
        value = float(sorted_p[start])
        stop = start + 1
        while stop < sorted_p.size and sorted_p[stop] == value:
            stop += 1
        count += stop - start
        overlap += int(sorted_a[start:stop].sum())
        next_value = float(sorted_p[stop]) if stop < sorted_p.size else 0.0
        denominator = count + action_size
        loss = 0.0 if denominator == 0 else 1.0 - 2.0 * overlap / denominator
        risk += (value - next_value) * loss
        start = stop
    return float(risk)


def quadrature_q_risks(probability, action, m_values=M_VALUES):
    results = {}
    action_size = int(action.sum())
    for m in m_values:
        dice_values = []
        nhd_values = []
        nhd95_values = []
        for node in (np.arange(m, dtype=float) + 0.5) / m:
            level = probability >= node
            denominator = int(level.sum()) + action_size
            if denominator == 0:
                dice_values.append(0.0)
            else:
                overlap = int(np.logical_and(level, action).sum())
                dice_values.append(1.0 - 2.0 * overlap / denominator)
            boundary = normalized_penalized_boundary_losses(action, level)
            nhd_values.append(boundary.nhd)
            nhd95_values.append(boundary.nhd95)
        results[m] = {
            "dice": float(np.mean(dice_values)),
            "nhd": float(np.mean(nhd_values)),
            "nhd95": float(np.mean(nhd95_values)),
        }
    return results


def _threshold_distribution(probability, coupling):
    """Enumerate the 1-D latent-mask law for Q or the antithetic coupling."""

    p = np.asarray(probability, dtype=float)
    if coupling == "shared_threshold":
        breaks = np.unique(np.concatenate(([0.0, 1.0], p.ravel())))
    elif coupling == "bimodal_antithetic":
        groups = _bimodal_groups(p.shape)
        breaks = np.unique(np.concatenate(([0.0, 1.0], p[groups], 1.0 - p[~groups])))
    else:
        raise ValueError("only one-dimensional latent laws can be enumerated")
    distribution = {}
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        mass = float(upper - lower)
        if mass <= 0:
            continue
        value = (float(lower) + float(upper)) / 2.0
        if coupling == "shared_threshold":
            mask = p >= value
        else:
            groups = _bimodal_groups(p.shape)
            mask = np.where(groups, p >= value, p >= 1.0 - value)
        key = np.packbits(mask.ravel()).tobytes()
        distribution[key] = distribution.get(key, 0.0) + mass
    return distribution


def _log_probability_target_independent(probability, target):
    p = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=bool)
    return float(np.log(p[target]).sum() + np.log1p(-p[~target]).sum())


def _probability_target_blocks(probability, target, block_size):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=bool)
    result = 1.0
    height, width = probability.shape
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            p = probability[top : top + block_size, left : left + block_size]
            included = target[top : top + block_size, left : left + block_size]
            upper = float(np.min(p[included])) if included.any() else 1.0
            lower = float(np.max(p[~included])) if (~included).any() else 0.0
            result *= max(0.0, upper - lower)
            if result == 0.0:
                return 0.0
    return float(result)


def exact_total_variation(probability, coupling, block_size):
    """Compute TV exactly by summing overlap on Q's finite support."""

    if coupling == "shared_threshold":
        return 0.0
    q_distribution = _threshold_distribution(probability, "shared_threshold")
    if coupling == "bimodal_antithetic":
        p_distribution = _threshold_distribution(probability, coupling)
        overlap = sum(
            min(q_mass, p_distribution.get(mask, 0.0))
            for mask, q_mass in q_distribution.items()
        )
    else:
        overlap = 0.0
        for packed, q_mass in q_distribution.items():
            target = (
                np.unpackbits(
                    np.frombuffer(packed, dtype=np.uint8), count=probability.size
                )
                .reshape(probability.shape)
                .astype(bool)
            )
            if coupling == "independent_bernoulli":
                log_mass = _log_probability_target_independent(probability, target)
                p_mass = 0.0 if log_mass < -745 else math.exp(log_mass)
            elif coupling == "local_block_threshold":
                p_mass = _probability_target_blocks(probability, target, block_size)
            else:
                raise ValueError(f"unknown coupling {coupling!r}")
            overlap += min(q_mass, p_mass)
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def exact_empty_probability(probability, coupling, block_size):
    p = np.asarray(probability, dtype=float)
    if coupling == "shared_threshold":
        return float(1.0 - np.max(p))
    if coupling == "independent_bernoulli":
        return float(math.exp(np.log1p(-p).sum()))
    if coupling == "local_block_threshold":
        result = 1.0
        for top in range(0, p.shape[0], block_size):
            for left in range(0, p.shape[1], block_size):
                block = p[top : top + block_size, left : left + block_size]
                result *= 1.0 - float(block.max())
        return float(result)
    if coupling == "bimodal_antithetic":
        groups = _bimodal_groups(p.shape)
        return float(max(0.0, 1.0 - float(p[groups].max()) - float(p[~groups].max())))
    raise ValueError(f"unknown coupling {coupling!r}")


def _loss_triplet(mask, action):
    boundary = normalized_penalized_boundary_losses(action, mask)
    return np.array([dice_loss(mask, action), boundary.nhd, boundary.nhd95])


def _finite_correlation(function, first, second):
    value = function(first, second).statistic
    return None if not np.isfinite(value) else float(value)


def _summarize(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _error_summary(approximation, truth):
    error = np.asarray(approximation) - np.asarray(truth)
    absolute = np.abs(error)
    result = _summarize(absolute)
    result["signed_bias"] = float(error.mean())
    return result


def _estimator_names(loss):
    names = [f"m{m}" for m in M_VALUES]
    if loss == "dice":
        names.append("exact")
    return tuple(names)


def _simulate_image(payload):
    probability, coupling, protocol, posterior_seed, image_index = payload
    draws = protocol["posterior_draws"]
    batches = protocol["mc_batches"]
    draws_per_batch = draws // batches
    action = probability >= protocol["gamma"]
    image_seed = _seed_from_parts(posterior_seed, image_index)
    random = np.random.default_rng(image_seed)
    global_uniforms = random.random(draws)
    p_masks, q_masks = sample_p_q(
        probability,
        coupling,
        global_uniforms,
        random,
        protocol["local_block_size"],
    )
    p_losses = np.empty((draws, len(LOSSES)))
    q_losses = np.empty_like(p_losses)
    jaccard_cost = np.empty(draws)
    nhd_cost = np.empty(draws)
    for draw_index in range(draws):
        p_mask = p_masks[draw_index]
        q_mask = q_masks[draw_index]
        p_losses[draw_index] = _loss_triplet(p_mask, action)
        q_losses[draw_index] = _loss_triplet(q_mask, action)
        jaccard_cost[draw_index] = jaccard_distance(p_mask, q_mask)
        nhd_cost[draw_index] = normalized_penalized_boundary_losses(p_mask, q_mask).nhd
    pushforward = {}
    for loss_index, loss in enumerate(LOSSES):
        per_draw = np.abs(
            np.sort(p_losses[:, loss_index]) - np.sort(q_losses[:, loss_index])
        )
        batch_values = np.abs(
            np.sort(p_losses[:, loss_index].reshape(batches, draws_per_batch), axis=1)
            - np.sort(q_losses[:, loss_index].reshape(batches, draws_per_batch), axis=1)
        ).mean(axis=1)
        pushforward[loss] = (
            float(per_draw.mean()),
            float(batch_values.std(ddof=1) / math.sqrt(batches)),
        )
    quadrature = quadrature_q_risks(probability, action)
    q_by_m = {m: np.array([quadrature[m][loss] for loss in LOSSES]) for m in M_VALUES}
    q_empty = 1.0 - float(probability.max())
    p_empty = exact_empty_probability(
        probability, coupling, protocol["local_block_size"]
    )
    return {
        "true_risk": p_losses.mean(axis=0),
        "q_mc_risk": q_losses.mean(axis=0),
        "true_mc_se": p_losses.std(axis=0, ddof=1) / math.sqrt(draws),
        "batch_risk": p_losses.reshape(batches, draws_per_batch, len(LOSSES)).mean(
            axis=1
        ),
        "q_by_m": q_by_m,
        "dice_exact": exact_dice_q_risk(probability, action),
        "action_fraction": float(action.mean()),
        "tv_exact": exact_total_variation(
            probability, coupling, protocol["local_block_size"]
        ),
        "empty_tv_lower": abs(p_empty - q_empty),
        "transport_jaccard": (
            float(jaccard_cost.mean()),
            float(jaccard_cost.std(ddof=1) / math.sqrt(draws)),
        ),
        "transport_nhd": (
            float(nhd_cost.mean()),
            float(nhd_cost.std(ddof=1) / math.sqrt(draws)),
        ),
        "pushforward": pushforward,
    }


def simulate_cell(spec, cell, *, workers=1):
    """Run one cell in memory and return aggregate sufficient statistics."""

    validate_cell(spec, cell)
    protocol = spec["protocol"]
    seeds = cell_seeds(spec, cell)
    cohort_size = protocol["cohort_size"]
    draws = protocol["posterior_draws"]
    batches = protocol["mc_batches"]
    map_rng = np.random.default_rng(seeds["map_seed"])
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    probabilities = [
        generate_probability_map(
            protocol, cell.morphology, cell.sharpness, map_rng, image_index
        )
        for image_index in range(cohort_size)
    ]

    true_risks = np.empty((cohort_size, len(LOSSES)))
    q_mc_risks = np.empty_like(true_risks)
    true_mc_se = np.empty_like(true_risks)
    batch_risks = np.empty((batches, cohort_size, len(LOSSES)))
    q_by_m = {m: np.empty_like(true_risks) for m in M_VALUES}
    dice_exact = np.empty(cohort_size)
    action_fraction = np.empty(cohort_size)
    tv_exact = np.empty(cohort_size)
    empty_tv_lower = np.empty(cohort_size)
    transport_jaccard = np.empty((cohort_size, 2))
    transport_nhd = np.empty((cohort_size, 2))
    pushforward = {loss: np.empty((cohort_size, 2)) for loss in LOSSES}

    payloads = [
        (probability, cell.coupling, protocol, seeds["posterior_seed"], image_index)
        for image_index, probability in enumerate(probabilities)
    ]
    if workers == 1:
        image_results = map(_simulate_image, payloads)
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, cohort_size))
        image_results = executor.map(_simulate_image, payloads)
    try:
        for image_index, result in enumerate(image_results):
            true_risks[image_index] = result["true_risk"]
            q_mc_risks[image_index] = result["q_mc_risk"]
            true_mc_se[image_index] = result["true_mc_se"]
            batch_risks[:, image_index, :] = result["batch_risk"]
            for m in M_VALUES:
                q_by_m[m][image_index] = result["q_by_m"][m]
            dice_exact[image_index] = result["dice_exact"]
            action_fraction[image_index] = result["action_fraction"]
            tv_exact[image_index] = result["tv_exact"]
            empty_tv_lower[image_index] = result["empty_tv_lower"]
            transport_jaccard[image_index] = result["transport_jaccard"]
            transport_nhd[image_index] = result["transport_nhd"]
            for loss in LOSSES:
                pushforward[loss][image_index] = result["pushforward"][loss]
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    loss_results = {}
    for loss_index, loss in enumerate(LOSSES):
        truth = true_risks[:, loss_index]
        estimators = {}
        for name in _estimator_names(loss):
            approximation = (
                dice_exact if name == "exact" else q_by_m[int(name[1:])][:, loss_index]
            )
            score = -approximation
            score_aurc = tie_aware_expected_aurc(score, truth)
            oracle_aurc = tie_aware_expected_aurc(-truth, truth)
            batch_regrets = []
            for batch in range(batches):
                batch_truth = batch_risks[batch, :, loss_index]
                batch_regrets.append(
                    tie_aware_expected_aurc(score, batch_truth)
                    - tie_aware_expected_aurc(-batch_truth, batch_truth)
                )
            estimators[name] = {
                "score_error": _error_summary(approximation, truth),
                "spearman_risk_ranking": _finite_correlation(
                    stats.spearmanr, approximation, truth
                ),
                "kendall_tau_b_risk_ranking": _finite_correlation(
                    stats.kendalltau, approximation, truth
                ),
                "aurc": float(score_aurc),
                "oracle_aurc": float(oracle_aurc),
                "aurc_regret": float(score_aurc - oracle_aurc),
                "aurc_regret_mc_se": float(
                    np.std(batch_regrets, ddof=1) / math.sqrt(batches)
                ),
            }
        cell_mean_mc_se = float(
            math.sqrt(np.square(true_mc_se[:, loss_index]).sum()) / cohort_size
        )
        loss_results[loss] = {
            "true_p_risk": _summarize(truth),
            "q_monte_carlo_risk": _summarize(q_mc_risks[:, loss_index]),
            "cell_mean_true_risk_mc_se": cell_mean_mc_se,
            "cell_mean_qmc_minus_true": float(
                (q_mc_risks[:, loss_index] - truth).mean()
            ),
            "estimators": estimators,
            "loss_pushforward_w1_empirical": {
                "mean": float(pushforward[loss][:, 0].mean()),
                "mean_mc_se": float(
                    math.sqrt(np.square(pushforward[loss][:, 1]).sum()) / cohort_size
                ),
                "maximum": float(pushforward[loss][:, 0].max()),
            },
        }

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "cell": {
            "coupling": cell.coupling,
            "sharpness": cell.sharpness,
            "morphology": cell.morphology,
            "replicate": cell.replicate,
        },
        "seeds": seeds,
        "cohort": {
            "size": cohort_size,
            "height": protocol["height"],
            "width": protocol["width"],
            "posterior_draws_per_image": draws,
            "mc_batches": batches,
            "action_foreground_fraction": _summarize(action_fraction),
        },
        "posterior_discrepancy": {
            "total_variation_exact": _summarize(tv_exact),
            "empty_event_tv_lower_bound_exact": _summarize(empty_tv_lower),
            "paired_jaccard_transport_cost_upper_bound": {
                "mean": float(transport_jaccard[:, 0].mean()),
                "mean_mc_se": float(
                    math.sqrt(np.square(transport_jaccard[:, 1]).sum()) / cohort_size
                ),
                "maximum": float(transport_jaccard[:, 0].max()),
            },
            "paired_normalized_hd_transport_cost_upper_bound": {
                "mean": float(transport_nhd[:, 0].mean()),
                "mean_mc_se": float(
                    math.sqrt(np.square(transport_nhd[:, 1]).sum()) / cohort_size
                ),
                "maximum": float(transport_nhd[:, 0].max()),
            },
            "interpretation": (
                "TV is exact for these constructed finite laws; paired transport costs "
                "are Monte Carlo upper bounds on optimal Jaccard/full-HD transport."
            ),
        },
        "losses": loss_results,
        "monte_carlo_note": (
            "True-P risks use independent posterior batches; reported MC SE concerns "
            "posterior integration only. Loss-pushforward W1 is empirical. nHD95 has "
            "no mask-Wasserstein corollary in this study."
        ),
    }


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _package_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run_cell(lock_binding, cell, phase, expected_cell_seed, command):
    spec = lock_binding["spec"]
    validate_cell(spec, cell)
    allowed = selected_cells(spec, phase)
    if cell not in set(allowed):
        raise ValueError(f"cell {cell.slug} is not part of phase {phase}")
    seeds = cell_seeds(spec, cell)
    if seeds["cell_seed"] != expected_cell_seed:
        raise ValueError("expected cell seed does not match the immutable derivation")
    root_key = "pilot_output_root" if phase == "pilot" else "main_output_root"
    output_root = _resolve_from_repo(lock_binding["path"], spec["paths"][root_key])
    identity = {
        "campaign_id": spec["campaign_id"],
        "lock_sha256": lock_binding["sha256"],
        "cell": cell.key,
        "seeds": seeds,
        "code_sources": [sha for _, sha in lock_binding["code_sources"]],
    }
    artifact_id = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    cell_directory = (
        output_root
        / cell.coupling
        / cell.sharpness
        / cell.morphology
        / (f"replicate-{cell.replicate:02d}")
    )
    final_directory = cell_directory / artifact_id
    if final_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite synthetic artifact {final_directory}"
        )
    cell_directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=cell_directory))
    start = time.monotonic()
    try:
        summary = simulate_cell(spec, cell, workers=spec["protocol"]["workers_per_job"])
        elapsed = time.monotonic() - start
        summary_path = temporary / "summary.json"
        _write_json(summary_path, summary)
        manifest = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "artifact_id": artifact_id,
            "campaign_id": spec["campaign_id"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "phase": phase,
            "cell": summary["cell"],
            "seeds": seeds,
            "lock": {
                "path": str(lock_binding["path"]),
                "sha256": lock_binding["sha256"],
            },
            "spec": {
                "path": str(lock_binding["spec_path"]),
                "sha256": lock_binding["spec_sha256"],
            },
            "code_sources": [
                {"path": str(path), "sha256": sha}
                for path, sha in lock_binding["code_sources"]
            ],
            "summary": {"path": "summary.json", "sha256": sha256_file(summary_path)},
            "runtime_seconds": float(elapsed),
            "environment": {
                "packages": _package_versions(),
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            },
            "command": list(command),
            "storage_policy": "aggregate sufficient statistics only; no masks persisted",
        }
        _write_json(temporary / "manifest.json", manifest)
        os.rename(temporary, final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_directory / "manifest.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    parser.add_argument("--coupling", choices=COUPLINGS, required=True)
    parser.add_argument("--sharpness", choices=SHARPNESS_LEVELS, required=True)
    parser.add_argument("--morphology", choices=MORPHOLOGIES, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--expected-cell-seed", type=int, required=True)
    arguments = list(argv) if argv is not None else None
    parsed = parser.parse_args(arguments)
    parsed.command = arguments if arguments is not None else os.sys.argv[1:]
    return parsed


def main(argv=None):
    args = parse_args(argv)
    binding = load_synthetic_lock(args.lock, expected_sha256=args.expected_lock_sha256)
    cell = Cell(args.coupling, args.sharpness, args.morphology, args.replicate)
    manifest = run_cell(
        binding, cell, args.phase, args.expected_cell_seed, args.command
    )
    print(manifest)


if __name__ == "__main__":
    main()
