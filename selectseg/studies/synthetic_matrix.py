"""Known-posterior coupling matrix for Dice, HD, and HD95.

One invocation evaluates one independent simulation cell.  The cell fixes the
true mask coupling, probability sharpness, morphology, and repeat, then scores
the same cohort with every declared estimator coupling.  This separates
working-posterior adequacy from loss geometry without modifying the immutable
v1 stress-test implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from selectseg.confidence import tie_aware_expected_aurc
from selectseg.counts import sample_spatial_copula_masks
from selectseg.quadrature import sha256_file
from selectseg.studies.synthetic import (
    COUPLINGS as TRUE_COUPLINGS,
    LOSSES,
    MORPHOLOGIES,
    SHARPNESS_LEVELS,
    Cell,
    _loss_triplet,
    _seed_from_parts,
    cell_seeds,
    generate_probability_map,
    sample_p_q,
)


ESTIMATOR_COUPLINGS = (
    "shared_threshold",
    "independent_bernoulli",
    "local_block_threshold",
    "spatial_copula",
)
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "selectseg.synthetic_coupling_matrix_cell"


def _load_config(path: Path) -> tuple[Path, dict]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"config must be a regular file: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported synthetic matrix config")
    if config.get("campaign_id") != "binary-synthetic-posterior-matrix-v1":
        raise ValueError("unexpected synthetic matrix campaign")
    grid = config.get("grid", {})
    expected_grid = {
        "true_couplings": list(TRUE_COUPLINGS),
        "estimator_couplings": list(ESTIMATOR_COUPLINGS),
        "sharpness_levels": list(SHARPNESS_LEVELS),
        "morphologies": list(MORPHOLOGIES),
        "replicates": 10,
    }
    if grid != expected_grid:
        raise ValueError("synthetic matrix grid differs from the declared design")
    protocol = config.get("protocol", {})
    required = {
        "height",
        "width",
        "cohort_size",
        "posterior_draws",
        "mc_batches",
        "workers_per_job",
        "gamma",
        "sharpness_pixels",
        "local_block_size",
        "losses",
        "spatial_copula",
    }
    if set(protocol) != required or protocol["losses"] != list(LOSSES):
        raise ValueError("synthetic matrix protocol has unexpected fields")
    if protocol["posterior_draws"] % protocol["mc_batches"]:
        raise ValueError("posterior draws must divide evenly into MC batches")
    if protocol["gamma"] != 0.5:
        raise ValueError("the synthetic matrix fixes the deployed action at 0.5")
    spatial = protocol["spatial_copula"]
    if set(spatial) != {
        "global_variance_weight",
        "spatial_variance_weight",
        "spatial_knot_spacing_diagonal",
        "posterior_batch_size",
    }:
        raise ValueError("invalid spatial-copula declaration")
    return path, config


def all_cells(config: dict) -> tuple[Cell, ...]:
    grid = config["grid"]
    return tuple(
        Cell(coupling, sharpness, morphology, repeat)
        for coupling in grid["true_couplings"]
        for sharpness in grid["sharpness_levels"]
        for morphology in grid["morphologies"]
        for repeat in range(grid["replicates"])
    )


def _finite_correlation(function, first, second):
    value = function(first, second).statistic
    return None if not np.isfinite(value) else float(value)


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "standard_deviation": (
            float(values.std(ddof=1)) if values.size > 1 else 0.0
        ),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _error_summary(estimate, truth):
    error = np.asarray(estimate, dtype=float) - np.asarray(truth, dtype=float)
    result = _summary(np.abs(error))
    result.update(
        {
            "root_mean_squared_error": float(np.sqrt(np.mean(error**2))),
            "signed_bias": float(error.mean()),
        }
    )
    return result


def _estimator_masks(
    probability,
    *,
    estimator,
    true_coupling,
    true_masks,
    shared_masks,
    uniforms,
    seed,
    protocol,
    image_index,
):
    if estimator == true_coupling:
        return true_masks
    if estimator == "shared_threshold":
        return shared_masks
    if estimator in {"independent_bernoulli", "local_block_threshold"}:
        random = np.random.default_rng(
            _seed_from_parts(seed, "estimator", estimator, image_index)
        )
        masks, _ = sample_p_q(
            probability,
            estimator,
            uniforms,
            random,
            protocol["local_block_size"],
        )
        return masks
    if estimator == "spatial_copula":
        spatial = protocol["spatial_copula"]
        masks, _ = sample_spatial_copula_masks(
            probability,
            posterior_draws=protocol["posterior_draws"],
            repeat_index=0,
            global_variance_weight=spatial["global_variance_weight"],
            spatial_variance_weight=spatial["spatial_variance_weight"],
            spatial_knot_spacing_diagonal=spatial[
                "spatial_knot_spacing_diagonal"
            ],
            posterior_batch_size=spatial["posterior_batch_size"],
            master_seed=seed,
            sample_id=f"synthetic-{image_index}",
            device="cpu",
        )
        return masks
    raise ValueError(f"unknown estimator coupling {estimator!r}")


def _simulate_image(payload):
    probability, cell, protocol, posterior_seed, image_index = payload
    draws = protocol["posterior_draws"]
    batches = protocol["mc_batches"]
    draws_per_batch = draws // batches
    action = probability >= protocol["gamma"]
    image_seed = _seed_from_parts(posterior_seed, image_index)
    random = np.random.default_rng(image_seed)
    uniforms = random.random(draws)
    true_masks, shared_masks = sample_p_q(
        probability,
        cell.coupling,
        uniforms,
        random,
        protocol["local_block_size"],
    )
    mask_sets = {
        estimator: _estimator_masks(
            probability,
            estimator=estimator,
            true_coupling=cell.coupling,
            true_masks=true_masks,
            shared_masks=shared_masks,
            uniforms=uniforms,
            seed=image_seed,
            protocol=protocol,
            image_index=image_index,
        )
        for estimator in ESTIMATOR_COUPLINGS
    }
    true_losses = np.empty((draws, len(LOSSES)))
    estimator_losses = {
        estimator: np.empty_like(true_losses) for estimator in ESTIMATOR_COUPLINGS
    }
    for draw in range(draws):
        true_losses[draw] = _loss_triplet(true_masks[draw], action)
        for estimator in ESTIMATOR_COUPLINGS:
            estimator_losses[estimator][draw] = _loss_triplet(
                mask_sets[estimator][draw], action
            )
    return {
        "true_risk": true_losses.mean(axis=0),
        "true_batches": true_losses.reshape(
            batches, draws_per_batch, len(LOSSES)
        ).mean(axis=1),
        "estimator_risk": {
            name: values.mean(axis=0) for name, values in estimator_losses.items()
        },
        "estimator_batches": {
            name: values.reshape(batches, draws_per_batch, len(LOSSES)).mean(axis=1)
            for name, values in estimator_losses.items()
        },
    }


def simulate_cell(config: dict, cell: Cell, *, workers: int | None = None) -> dict:
    if cell not in set(all_cells(config)):
        raise ValueError("cell is outside the declared synthetic matrix")
    protocol = config["protocol"]
    workers = protocol["workers_per_job"] if workers is None else workers
    if workers < 1:
        raise ValueError("workers must be positive")
    seeds = cell_seeds(
        {
            "base_seed": config["base_seed"],
        },
        cell,
    )
    map_random = np.random.default_rng(seeds["map_seed"])
    probabilities = [
        generate_probability_map(
            protocol,
            cell.morphology,
            cell.sharpness,
            map_random,
            image_index,
        )
        for image_index in range(protocol["cohort_size"])
    ]
    payloads = [
        (probability, cell, protocol, seeds["posterior_seed"], image_index)
        for image_index, probability in enumerate(probabilities)
    ]
    if workers == 1:
        image_results = list(map(_simulate_image, payloads))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            image_results = list(executor.map(_simulate_image, payloads))

    truth = np.stack([result["true_risk"] for result in image_results])
    true_batches = np.stack([result["true_batches"] for result in image_results], axis=1)
    estimator_risks = {
        estimator: np.stack(
            [result["estimator_risk"][estimator] for result in image_results]
        )
        for estimator in ESTIMATOR_COUPLINGS
    }
    estimator_batches = {
        estimator: np.stack(
            [result["estimator_batches"][estimator] for result in image_results],
            axis=1,
        )
        for estimator in ESTIMATOR_COUPLINGS
    }

    losses = {}
    for loss_index, loss in enumerate(LOSSES):
        true_loss = truth[:, loss_index]
        oracle_aurc = tie_aware_expected_aurc(-true_loss, true_loss)
        estimators = {}
        for estimator in ESTIMATOR_COUPLINGS:
            estimate = estimator_risks[estimator][:, loss_index]
            score_aurc = tie_aware_expected_aurc(-estimate, true_loss)
            batch_differences = (
                estimator_batches[estimator][:, :, loss_index]
                - true_batches[:, :, loss_index]
            ).mean(axis=1)
            estimators[estimator] = {
                "risk_error": _error_summary(estimate, true_loss),
                "spearman_risk_ranking": _finite_correlation(
                    stats.spearmanr, estimate, true_loss
                ),
                "kendall_tau_b_risk_ranking": _finite_correlation(
                    stats.kendalltau, estimate, true_loss
                ),
                "aurc": float(score_aurc),
                "oracle_aurc": float(oracle_aurc),
                "aurc_regret": float(score_aurc - oracle_aurc),
                "cell_mean_risk_difference_mc_se": float(
                    batch_differences.std(ddof=1) / math.sqrt(protocol["mc_batches"])
                ),
            }
        losses[loss] = {
            "true_risk": _summary(true_loss),
            "estimators": estimators,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "cell": {
            "true_coupling": cell.coupling,
            "sharpness": cell.sharpness,
            "morphology": cell.morphology,
            "replicate": cell.replicate,
        },
        "seeds": seeds,
        "cohort": {
            "size": protocol["cohort_size"],
            "height": protocol["height"],
            "width": protocol["width"],
            "posterior_draws": protocol["posterior_draws"],
            "mc_batches": protocol["mc_batches"],
        },
        "losses": losses,
    }


def _source_bindings() -> list[dict]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        "selectseg/studies/synthetic_matrix.py",
        "selectseg/studies/synthetic.py",
        "selectseg/counts.py",
        "selectseg/geometry.py",
        "selectseg/confidence.py",
    )
    return [
        {"path": path, "sha256": sha256_file(root / path)}
        for path in paths
    ]


def run_cell(config_path: Path, config: dict, cell: Cell, command) -> Path:
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    config_sha = sha256_file(config_path)
    sources = _source_bindings()
    identity = {
        "campaign_id": config["campaign_id"],
        "config_sha256": config_sha,
        "cell": cell.key,
        "source_sha256": [entry["sha256"] for entry in sources],
    }
    artifact_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    cell_root = (
        output_root
        / cell.coupling
        / cell.sharpness
        / cell.morphology
        / f"replicate-{cell.replicate:02d}"
    )
    final = cell_root / artifact_id
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    cell_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=cell_root))
    started = time.monotonic()
    try:
        summary = simulate_cell(config, cell)
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "artifact_id": artifact_id,
            "campaign_id": config["campaign_id"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cell": summary["cell"],
            "config": {"path": str(config_path), "sha256": config_sha},
            "code_sources": sources,
            "summary": {
                "path": "summary.json",
                "sha256": sha256_file(summary_path),
            },
            "runtime_seconds": float(time.monotonic() - started),
            "environment": {
                "python": platform.python_version(),
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "command": list(command),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final / "manifest.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--true-coupling", choices=TRUE_COUPLINGS, required=True)
    parser.add_argument("--sharpness", choices=SHARPNESS_LEVELS, required=True)
    parser.add_argument("--morphology", choices=MORPHOLOGIES, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--expected-cell-seed", type=int, required=True)
    arguments = list(argv) if argv is not None else os.sys.argv[1:]
    args = parser.parse_args(arguments)
    args.command = arguments
    return args


def main(argv=None):
    args = parse_args(argv)
    config_path, config = _load_config(Path(args.config))
    if sha256_file(config_path) != args.expected_config_sha256:
        raise ValueError("synthetic matrix config SHA-256 mismatch")
    cell = Cell(
        args.true_coupling,
        args.sharpness,
        args.morphology,
        args.replicate,
    )
    expected = cell_seeds({"base_seed": config["base_seed"]}, cell)["cell_seed"]
    if expected != args.expected_cell_seed:
        raise ValueError("synthetic matrix cell seed mismatch")
    print(run_cell(config_path, config, cell, args.command))


if __name__ == "__main__":
    main()
