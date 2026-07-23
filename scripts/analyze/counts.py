"""Analyze predeclared Dice count and spatial-partition experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

from scripts.analyze.main import paired_cluster_bootstrap_aurc_test
from scripts.analyze.ensemble import _load_artifact, _one_manifest
from selectseg.artifacts import sha256_file
from selectseg.confidence import summarize_aurc


MAIN_ARTIFACT_TYPE = "selectseg.binary_simulation_assembly"
COUNT_ARTIFACT_TYPE = "selectseg.binary_dice_count_posterior"
PARTITION_ARTIFACT_TYPE = "selectseg.binary_dice_partition_posterior"
COPULA_ARTIFACT_TYPE = "selectseg.binary_dice_spatial_copula"
METHODS = (
    ("confidence_dice_action_two_block_m32", "Dice two-block"),
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_dice_exact", "Dice-Exact"),
    ("confidence_sdc", "SDC"),
    ("confidence_foreground_entropy", "Foreground entropy"),
)
CONTROL_ABSOLUTE_TOLERANCE = 4 * np.finfo(np.float64).eps
SDC_SIZE_STRATA = (
    ("empty", 0, 0),
    ("very_small", 1, 31),
    ("small", 32, 255),
    ("medium", 256, 4095),
    ("large", 4096, None),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("count", "partition", "copula"), default="count"
    )
    parser.add_argument(
        "--contract",
        default="configs/auxiliary/dice_count_result_analysis_v1.json",
    )
    parser.add_argument("--main-root", default="outputs/binary_midpoint_main_v2/assembled")
    parser.add_argument("--count-root", default="outputs/dice_count_posterior_v1")
    parser.add_argument("--partition-root", default="outputs/dice_partition_ladder_v1")
    parser.add_argument("--copula-root", default="outputs/spatial_copula_v1")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _rank_metrics(score: np.ndarray, risk: np.ndarray) -> dict:
    if np.unique(score).size < 2 or np.unique(risk).size < 2:
        return {"spearman_rho_with_safety": None, "kendall_tau_b_with_safety": None}
    safety = -risk
    spearman = float(
        np.corrcoef(rankdata(score, method="average"), rankdata(safety, method="average"))[0, 1]
    )
    kendall = float(kendalltau(score, safety, variant="b", method="auto").statistic)
    if not math.isfinite(spearman) or not math.isfinite(kendall):
        raise RuntimeError("rank metric became non-finite")
    return {"spearman_rho_with_safety": spearman, "kendall_tau_b_with_safety": kendall}


def _derived_seed(base: int, dataset: str, condition: str) -> int:
    value = f"{base}|{dataset}|{condition}|dice-count".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _partition_seed(base: int, dataset: str, condition: str, *parts: str) -> int:
    value = "|".join((str(base), dataset, condition, *parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _variant_id(variant: dict) -> str:
    coupling = variant["coupling"]
    threshold = variant["proposal_threshold"]
    if threshold is None:
        return coupling
    return f"{coupling}_t{int(round(100 * threshold)):02d}"


def _partition_manifest(
    root: Path, dataset: str, condition: str, variant: str
) -> Path:
    candidates = sorted((root / dataset / condition / variant).glob("*/manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {variant} artifact for {dataset}/{condition} under {root}"
        )
    return candidates[0]


def _score_summary(score: np.ndarray, risk: np.ndarray, label: str) -> dict:
    return {
        "label": label,
        **asdict(summarize_aurc(score, risk)),
        **_rank_metrics(score, risk),
    }


def _score_agreement(left: np.ndarray, right: np.ndarray) -> dict:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return {"spearman_rho": None, "kendall_tau_b": None}
    spearman = float(
        np.corrcoef(
            rankdata(left, method="average"),
            rankdata(right, method="average"),
        )[0, 1]
    )
    kendall = float(kendalltau(left, right, variant="b", method="auto").statistic)
    if not math.isfinite(spearman) or not math.isfinite(kendall):
        raise RuntimeError("score agreement became non-finite")
    return {"spearman_rho": spearman, "kendall_tau_b": kendall}


def _finite_summary(values: np.ndarray) -> dict:
    """Summarize one finite numeric vector without hiding its upper tail."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summary input must be a nonempty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def sdc_bound_audit(rows: list[dict]) -> dict:
    """Audit the count-dispersion bound against exact level-set Dice.

    For nonempty actions, the theorem gives

    ``error <= 2 sqrt(Var Z) / m + sqrt(Var W) / (2m)``.

    Empty actions are reported separately because the theorem's ``1/m`` terms
    are undefined there.  The calculation uses Exact Dice from the main
    artifact and the deterministic midpoint count variances already stored in
    the count artifact, so no posterior Monte Carlo is introduced.
    """

    if not rows:
        raise ValueError("SDC audit requires at least one joined row")
    records = []
    empty_count = 0
    for row in rows:
        action_size = int(row["action_pixels"])
        if action_size < 0:
            raise ValueError("action_pixels must be nonnegative")
        if action_size == 0:
            empty_count += 1
            continue
        exact_risk = -float(row["confidence_dice_exact"])
        plug_in_risk = 1.0 - float(row["confidence_dice_sdc_recomputed"])
        error = abs(exact_risk - plug_in_risk)
        overlap_term = (
            2.0 * math.sqrt(float(row["shared_variance_overlap"])) / action_size
        )
        outside_term = (
            math.sqrt(float(row["shared_variance_outside"]))
            / (2.0 * action_size)
        )
        bound = overlap_term + outside_term
        slack = bound - error
        tolerance = 64 * np.finfo(np.float64).eps * max(1.0, bound, error)
        records.append(
            {
                "action_size": action_size,
                "error": error,
                "bound": bound,
                "slack": slack,
                "ratio": 0.0 if bound == 0 and error == 0 else error / bound,
                "overlap_term": overlap_term,
                "outside_term": outside_term,
                "holds": slack >= -tolerance,
            }
        )
    if not records:
        return {
            "num_images": len(rows),
            "num_nonempty_actions": 0,
            "num_empty_actions": empty_count,
            "all_nonempty_bounds_hold": True,
            "maximum_numerical_violation": 0.0,
            "strata": [],
        }

    def summarize(selected: list[dict], label: str) -> dict:
        error = np.asarray([record["error"] for record in selected])
        bound = np.asarray([record["bound"] for record in selected])
        slack = np.asarray([record["slack"] for record in selected])
        ratio = np.asarray([record["ratio"] for record in selected])
        overlap = np.asarray([record["overlap_term"] for record in selected])
        outside = np.asarray([record["outside_term"] for record in selected])
        return {
            "stratum": label,
            "num_images": len(selected),
            "action_pixels": {
                "minimum": min(record["action_size"] for record in selected),
                "maximum": max(record["action_size"] for record in selected),
            },
            "absolute_sdc_risk_error": _finite_summary(error),
            "theoretical_bound": _finite_summary(bound),
            "bound_slack": _finite_summary(slack),
            "error_to_bound_ratio": _finite_summary(ratio),
            "mean_overlap_term": float(overlap.mean()),
            "mean_outside_term": float(outside.mean()),
        }

    strata = []
    for label, lower, upper in SDC_SIZE_STRATA[1:]:
        selected = [
            record
            for record in records
            if record["action_size"] >= lower
            and (upper is None or record["action_size"] <= upper)
        ]
        if selected:
            strata.append(summarize(selected, label))
    violations = [max(0.0, -record["slack"]) for record in records]
    return {
        "num_images": len(rows),
        "num_nonempty_actions": len(records),
        "num_empty_actions": empty_count,
        "all_nonempty_bounds_hold": all(record["holds"] for record in records),
        "maximum_numerical_violation": float(max(violations)),
        "overall": summarize(records, "all_nonempty"),
        "strata": strata,
        "interpretation": (
            "The bound is explanatory rather than expected to be tight; its "
            "1/action-size factors can be conservative for small predictions."
        ),
    }


def _validate_count_controls(main: dict, auxiliary: dict) -> None:
    equality = {
        "risk_dice": "risk_dice",
        "confidence_dice_shared_m32_recomputed": "confidence_dice_m32",
        "confidence_dice_sdc_recomputed": "confidence_sdc",
        "image_index": "image_index",
    }
    for auxiliary_field, main_field in equality.items():
        exact = main_field in {"risk_dice", "image_index"}
        agrees = (
            auxiliary[auxiliary_field] == main[main_field]
            if exact
            else math.isclose(
                auxiliary[auxiliary_field],
                main[main_field],
                rel_tol=0.0,
                abs_tol=CONTROL_ABSOLUTE_TOLERANCE,
            )
        )
        if not agrees:
            raise ValueError(f"locked control mismatch for {main['sample_id']}/{main_field}")


def analyze(
    contract_path: Path,
    main_root: Path,
    count_root: Path,
    *,
    bootstrap_resamples: int | None = None,
) -> dict:
    contract = _load_json(contract_path)
    if contract.get("status") != "predeclared-before-reading-dice-count-posterior-outputs":
        raise ValueError("Dice count result contract has an unexpected status")
    score_contract = Path(contract["score_contract"]["path"])
    if sha256_file(score_contract) != contract["score_contract"]["sha256"]:
        raise ValueError("bound Dice count scoring contract changed")
    expected = [
        (dataset, model)
        for dataset in contract["conditions"]["datasets"]
        for model in contract["conditions"]["models"]
    ]
    if len(expected) != contract["conditions"]["count"] or len(expected) != 10:
        raise ValueError("Dice count result contract must identify ten conditions")
    n_resamples = (
        contract["reporting"]["bootstrap_resamples"]
        if bootstrap_resamples is None
        else bootstrap_resamples
    )
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    base_seed = contract["reporting"]["bootstrap_seed"]
    primary = contract["primary_comparison"]
    conditions = []
    differences = []
    for dataset, condition in sorted(expected):
        main_path = _one_manifest(main_root, dataset, condition)
        count_path = _one_manifest(count_root, dataset, condition)
        _, main_rows = _load_artifact(main_path, MAIN_ARTIFACT_TYPE)
        count_manifest, count_rows = _load_artifact(count_path, COUNT_ARTIFACT_TYPE)
        if count_manifest["analysis_contract_sha256"] != contract["score_contract"]["sha256"]:
            raise ValueError("count artifact is not bound to the declared score contract")
        counts_by_id = {row["sample_id"]: row for row in count_rows}
        if set(counts_by_id) != {row["sample_id"] for row in main_rows}:
            raise ValueError("main and count artifacts contain different cohorts")
        rows = []
        for main in main_rows:
            auxiliary = counts_by_id[main["sample_id"]]
            _validate_count_controls(main, auxiliary)
            rows.append({**main, **{k: v for k, v in auxiliary.items() if k not in main}})
        risk = np.asarray([row["risk_dice"] for row in rows], dtype=float)
        method_results = {}
        for field, label in METHODS:
            score = np.asarray([row[field] for row in rows], dtype=float)
            method_results[field] = {
                "label": label,
                **asdict(summarize_aurc(score, risk)),
                **_rank_metrics(score, risk),
            }
        left = np.asarray([row[primary["left"]] for row in rows], dtype=float)
        right = np.asarray([row[primary["right"]] for row in rows], dtype=float)
        sample_ids = [row["sample_id"] for row in rows]
        bootstrap = paired_cluster_bootstrap_aurc_test(
            left,
            right,
            risk,
            cluster_ids=sample_ids,
            n_resamples=n_resamples,
            seed=_derived_seed(base_seed, dataset, condition),
        )
        differences.append(bootstrap.difference)
        score_delta = left - right
        covariance = np.asarray([row["shared_covariance"] for row in rows], dtype=float)
        correlation = None
        if np.unique(score_delta).size > 1 and np.unique(covariance).size > 1:
            correlation = float(
                np.corrcoef(
                    rankdata(score_delta, method="average"),
                    rankdata(covariance, method="average"),
                )[0, 1]
            )
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "num_images": len(rows),
                "main_manifest_sha256": sha256_file(main_path),
                "count_manifest_sha256": sha256_file(count_path),
                "methods": method_results,
                "primary_comparison": {
                    "left": primary["left"],
                    "right": primary["right"],
                    "difference_left_minus_right": bootstrap.difference,
                    "bootstrap": asdict(bootstrap),
                },
                "mechanism": {
                    "mean_shared_count_covariance": float(covariance.mean()),
                    "mean_two_block_minus_shared_score": float(score_delta.mean()),
                    "spearman_shared_covariance_vs_score_delta": correlation,
                    "total_score_runtime_seconds": float(
                        sum(row["score_runtime_seconds"] for row in rows)
                    ),
                },
                "sdc_bound_audit": sdc_bound_audit(rows),
            }
        )
    array = np.asarray(differences)
    gate = contract["interpretation_gate"]
    summary = {
        "mean_difference_two_block_minus_shared": float(array.mean()),
        "median_difference_two_block_minus_shared": float(np.median(array)),
        "two_block_wins": int(np.count_nonzero(array < 0)),
        "shared_wins": int(np.count_nonzero(array > 0)),
        "ties": int(np.count_nonzero(array == 0)),
    }
    summary["interpretation_gate_passed"] = bool(
        summary["two_block_wins"] >= gate["required_condition_wins"]
        and summary["mean_difference_two_block_minus_shared"] < 0
    )
    return {
        "schema_version": 1,
        "analysis_id": contract["analysis_id"],
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "analysis": {
            "tie_policy": contract["reporting"]["tie_policy"],
            "bootstrap_resamples": n_resamples,
            "aurc_scale_for_display": contract["reporting"]["aurc_scale"],
            "control_validation": {
                "risk_and_image_index": "exact equality",
                "recomputed_scores_absolute_tolerance": CONTROL_ABSOLUTE_TOLERANCE,
                "observed_preanalysis_maximum_difference": 2.220446049250313e-16,
            },
        },
        "conditions": conditions,
        "summary": summary,
    }


def analyze_partitions(
    contract_path: Path,
    main_root: Path,
    count_root: Path,
    partition_root: Path,
    *,
    bootstrap_resamples: int | None = None,
) -> dict:
    """Analyze every declared component partition against fixed controls."""

    contract = _load_json(contract_path)
    if contract.get("status") != "predeclared-before-computing-component-or-grid-scores":
        raise ValueError("Dice partition contract has an unexpected status")
    parent = contract["parent_contract"]
    if sha256_file(Path(parent["path"])) != parent["sha256"]:
        raise ValueError("bound parent Dice coupling contract changed")
    expected = [
        (dataset, model)
        for dataset in contract["conditions"]["datasets"]
        for model in contract["conditions"]["models"]
    ]
    if len(expected) != contract["conditions"]["count"] or len(expected) != 10:
        raise ValueError("Dice partition contract must identify ten conditions")
    variants = {_variant_id(value): value for value in contract["variants"]}
    if len(variants) != 8:
        raise ValueError("Dice partition contract must identify eight variants")
    component_variants = [variant for variant in variants if "components" in variant]
    if len(component_variants) != 4:
        raise ValueError("Dice partition contract must identify four component variants")
    n_resamples = (
        contract["reporting"]["bootstrap_resamples"]
        if bootstrap_resamples is None
        else bootstrap_resamples
    )
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    contract_sha256 = sha256_file(contract_path)
    parent_sha256 = parent["sha256"]
    base_seed = contract["numerics"]["master_seed"]
    conditions = []
    differences = {
        variant: {"two_block": [], "matched_grid": []}
        for variant in component_variants
    }
    baseline_labels = {
        "levelset": "LevelSet-Q Dice",
        "two_block": "Dice two-block",
        "sdc": "SDC",
        "foreground_entropy": "Foreground entropy",
    }

    for dataset, condition in sorted(expected):
        main_path = _one_manifest(main_root, dataset, condition)
        count_path = _one_manifest(count_root, dataset, condition)
        _, main_rows = _load_artifact(main_path, MAIN_ARTIFACT_TYPE)
        count_manifest, count_rows = _load_artifact(count_path, COUNT_ARTIFACT_TYPE)
        if count_manifest["analysis_contract_sha256"] != parent_sha256:
            raise ValueError("two-block artifact is not bound to the parent contract")
        main_by_id = {row["sample_id"]: row for row in main_rows}
        count_by_id = {row["sample_id"]: row for row in count_rows}
        if set(main_by_id) != set(count_by_id):
            raise ValueError("main and two-block artifacts contain different cohorts")
        sample_ids = list(main_by_id)
        for sample_id in sample_ids:
            main = main_by_id[sample_id]
            auxiliary = count_by_id[sample_id]
            _validate_count_controls(main, auxiliary)
        risk = np.asarray([main_by_id[key]["risk_dice"] for key in sample_ids])
        scores = {
            "levelset": np.asarray(
                [main_by_id[key]["confidence_dice_m32"] for key in sample_ids]
            ),
            "two_block": np.asarray(
                [
                    count_by_id[key]["confidence_dice_action_two_block_m32"]
                    for key in sample_ids
                ]
            ),
            "sdc": np.asarray(
                [main_by_id[key]["confidence_sdc"] for key in sample_ids]
            ),
            "foreground_entropy": np.asarray(
                [
                    main_by_id[key]["confidence_foreground_entropy"]
                    for key in sample_ids
                ]
            ),
        }
        manifests = {}
        diagnostics = {}
        for variant, declaration in variants.items():
            manifest_path = _partition_manifest(
                partition_root, dataset, condition, variant
            )
            manifest, rows = _load_artifact(manifest_path, PARTITION_ARTIFACT_TYPE)
            if manifest["analysis_contract_sha256"] != contract_sha256:
                raise ValueError(f"{variant} artifact is not bound to the contract")
            expected_coupling = declaration["coupling"].replace("_", "-")
            if (
                manifest["coupling"] != expected_coupling
                or manifest["proposal_threshold"] != declaration["proposal_threshold"]
            ):
                raise ValueError(f"{variant} manifest does not match its declaration")
            rows_by_id = {row["sample_id"]: row for row in rows}
            if set(rows_by_id) != set(main_by_id):
                raise ValueError(f"{variant} artifact contains a different cohort")
            for sample_id in sample_ids:
                row = rows_by_id[sample_id]
                main = main_by_id[sample_id]
                if row["risk_dice"] != main["risk_dice"]:
                    raise ValueError(f"{variant} risk mismatch for {sample_id}")
                if row["image_index"] != main["image_index"]:
                    raise ValueError(f"{variant} image-index mismatch for {sample_id}")
            scores[variant] = np.asarray(
                [rows_by_id[key]["confidence_dice_partition"] for key in sample_ids]
            )
            repeat_scores = np.asarray(
                [rows_by_id[key]["repeat_confidences"] for key in sample_ids],
                dtype=float,
            )
            expected_shape = (len(sample_ids), contract["numerics"]["monte_carlo_repeats"])
            if repeat_scores.shape != expected_shape or not np.isfinite(repeat_scores).all():
                raise ValueError(f"{variant} contains invalid repeat confidences")
            repeat_aurcs = [
                summarize_aurc(repeat_scores[:, repeat], risk).aurc
                for repeat in range(repeat_scores.shape[1])
            ]
            repeat_agreement = [
                _score_agreement(repeat_scores[:, repeat], scores[variant])
                for repeat in range(repeat_scores.shape[1])
            ]
            manifests[variant] = sha256_file(manifest_path)
            diagnostics[variant] = {
                "mean_num_blocks": float(
                    np.mean([rows_by_id[key]["num_blocks"] for key in sample_ids])
                ),
                "mean_largest_block_fraction": float(
                    np.mean(
                        [
                            rows_by_id[key]["largest_block_fraction"]
                            for key in sample_ids
                        ]
                    )
                ),
                "mean_monte_carlo_repeat_standard_deviation": float(
                    np.mean(
                        [
                            rows_by_id[key]["monte_carlo_repeat_standard_deviation"]
                            for key in sample_ids
                        ]
                    )
                ),
                "total_score_runtime_seconds": float(
                    sum(rows_by_id[key]["score_runtime_seconds"] for key in sample_ids)
                ),
                "repeat_aurcs": repeat_aurcs,
                "repeat_aurc_range": float(max(repeat_aurcs) - min(repeat_aurcs)),
                "minimum_repeat_to_mean_spearman": min(
                    value["spearman_rho"] for value in repeat_agreement
                ),
            }

        method_results = {
            name: _score_summary(scores[name], risk, label)
            for name, label in baseline_labels.items()
        }
        method_results.update(
            {
                variant: _score_summary(
                    scores[variant],
                    risk,
                    variant.replace("_", " "),
                )
                for variant in variants
            }
        )
        comparisons = {}
        for component in sorted(component_variants):
            grid = component.replace("components", "grid")
            component_comparisons = {}
            for reference in ("two_block", grid):
                bootstrap = paired_cluster_bootstrap_aurc_test(
                    scores[component],
                    scores[reference],
                    risk,
                    cluster_ids=sample_ids,
                    n_resamples=n_resamples,
                    seed=_partition_seed(
                        base_seed,
                        dataset,
                        condition,
                        component,
                        reference,
                    ),
                )
                key = "matched_grid" if reference == grid else reference
                component_comparisons[key] = {
                    "reference": reference,
                    "difference_component_minus_reference": bootstrap.difference,
                    "bootstrap": asdict(bootstrap),
                    "score_agreement": _score_agreement(
                        scores[component], scores[reference]
                    ),
                }
                differences[component][key].append(bootstrap.difference)
            component_comparisons["difference_component_minus_sdc"] = float(
                method_results[component]["aurc"] - method_results["sdc"]["aurc"]
            )
            component_comparisons["difference_component_minus_entropy"] = float(
                method_results[component]["aurc"]
                - method_results["foreground_entropy"]["aurc"]
            )
            comparisons[component] = component_comparisons
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "num_images": len(sample_ids),
                "main_manifest_sha256": sha256_file(main_path),
                "count_manifest_sha256": sha256_file(count_path),
                "partition_manifest_sha256": manifests,
                "methods": method_results,
                "comparisons": comparisons,
                "diagnostics": diagnostics,
            }
        )

    summary = {}
    for component in sorted(component_variants):
        against_two = np.asarray(differences[component]["two_block"])
        against_grid = np.asarray(differences[component]["matched_grid"])
        summary[component] = {
            "component_minus_two_block": {
                "mean": float(against_two.mean()),
                "median": float(np.median(against_two)),
                "wins": int(np.count_nonzero(against_two < 0)),
                "losses": int(np.count_nonzero(against_two > 0)),
                "ties": int(np.count_nonzero(against_two == 0)),
            },
            "component_minus_matched_grid": {
                "mean": float(against_grid.mean()),
                "median": float(np.median(against_grid)),
                "wins": int(np.count_nonzero(against_grid < 0)),
                "losses": int(np.count_nonzero(against_grid > 0)),
                "ties": int(np.count_nonzero(against_grid == 0)),
            },
        }
        summary[component]["directional_gate_passed"] = bool(
            summary[component]["component_minus_two_block"]["wins"] >= 7
            and summary[component]["component_minus_matched_grid"]["wins"] >= 7
            and summary[component]["component_minus_two_block"]["mean"] < 0
            and summary[component]["component_minus_matched_grid"]["mean"] < 0
        )
    return {
        "schema_version": 1,
        "analysis_id": contract["analysis_id"],
        "contract": {"path": str(contract_path), "sha256": contract_sha256},
        "analysis": {
            "tie_policy": contract["reporting"]["tie_policy"],
            "bootstrap_resamples": n_resamples,
            "aurc_scale_for_display": contract["reporting"]["aurc_scale"],
            "selection_policy": "all declared variants reported; no test-set selection",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "conditions": conditions,
        "summary": summary,
    }


def _copula_variant_id(manifest: dict, variants: dict[str, dict]) -> str:
    parameters = manifest.get("spatial_copula")
    if not isinstance(parameters, dict):
        raise ValueError("spatial-copula manifest lacks its parameter block")
    matches = []
    for variant_id, variant in variants.items():
        if (
            parameters.get("global_variance_weight")
            == variant["global_variance_weight"]
            and parameters.get("spatial_variance_weight")
            == variant["spatial_variance_weight"]
            and parameters.get("spatial_knot_spacing_diagonal")
            == variant["spatial_knot_spacing_diagonal"]
        ):
            matches.append(variant_id)
    if len(matches) != 1:
        raise ValueError(
            "spatial-copula manifest matches "
            f"{len(matches)} declared variants instead of one"
        )
    return matches[0]


def analyze_spatial_copula(
    contract_path: Path,
    main_root: Path,
    copula_root: Path,
    *,
    bootstrap_resamples: int | None = None,
) -> dict:
    """Aggregate independent copula repeats and evaluate every declared variant."""

    contract = _load_json(contract_path)
    if contract.get("status") != "predeclared-before-reading-spatial-copula-outputs":
        raise ValueError("spatial-copula analysis contract has an unexpected status")
    score_binding = contract["score_contract"]
    score_contract_path = Path(score_binding["path"])
    if sha256_file(score_contract_path) != score_binding["sha256"]:
        raise ValueError("bound spatial-copula scoring contract changed")
    score_contract = _load_json(score_contract_path)
    if score_contract.get("status") != "predeclared-before-computing-spatial-copula-scores":
        raise ValueError("bound scoring contract is not a spatial-copula contract")

    expected = {
        (dataset, condition)
        for dataset in contract["conditions"]["datasets"]
        for condition in contract["conditions"]["models"]
    }
    if (
        len(expected) != contract["conditions"]["count"]
        or expected
        != {
            (dataset, condition)
            for dataset in score_contract["conditions"]["datasets"]
            for condition in score_contract["conditions"]["models"]
        }
    ):
        raise ValueError("analysis and scoring contracts identify different conditions")
    variants = {variant["id"]: variant for variant in score_contract["variants"]}
    if len(variants) != len(score_contract["variants"]):
        raise ValueError("spatial-copula variant ids are not unique")
    repeat_indices = score_contract["numerics"]["repeat_indices"]
    n_resamples = (
        contract["reporting"]["bootstrap_resamples"]
        if bootstrap_resamples is None
        else bootstrap_resamples
    )
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    contract_sha256 = sha256_file(contract_path)
    score_contract_sha256 = score_binding["sha256"]
    primary_variant = contract["primary_comparison"]["variant_id"]
    if primary_variant not in variants:
        raise ValueError("primary spatial-copula variant is not declared")

    conditions = []
    primary_differences = []
    for dataset, condition in sorted(expected):
        main_path = _one_manifest(main_root, dataset, condition)
        _, main_rows = _load_artifact(main_path, MAIN_ARTIFACT_TYPE)
        main_by_id = {row["sample_id"]: row for row in main_rows}
        sample_ids = [row["sample_id"] for row in main_rows]
        risk = np.asarray([row["risk_dice"] for row in main_rows], dtype=float)

        manifest_paths = sorted(
            (copula_root / dataset / condition / "spatial_copula").glob(
                "*/manifest.json"
            )
        )
        expected_artifacts = len(variants) * len(repeat_indices)
        if len(manifest_paths) != expected_artifacts:
            raise ValueError(
                f"expected {expected_artifacts} spatial-copula artifacts for "
                f"{dataset}/{condition}, found {len(manifest_paths)}"
            )
        repeat_scores: dict[str, dict[int, np.ndarray]] = {
            variant_id: {} for variant_id in variants
        }
        repeat_runtime: dict[str, dict[int, float]] = {
            variant_id: {} for variant_id in variants
        }
        manifest_sha256: dict[str, dict[int, str]] = {
            variant_id: {} for variant_id in variants
        }
        source_sha256 = set()
        for manifest_path in manifest_paths:
            manifest, rows = _load_artifact(manifest_path, COPULA_ARTIFACT_TYPE)
            if manifest["analysis_contract_sha256"] != score_contract_sha256:
                raise ValueError("copula artifact is bound to another score contract")
            variant_id = _copula_variant_id(manifest, variants)
            parameters = manifest["spatial_copula"]
            repeat_index = parameters["repeat_index"]
            if repeat_index not in repeat_indices:
                raise ValueError("copula artifact has an undeclared repeat index")
            if repeat_index in repeat_scores[variant_id]:
                raise ValueError("duplicate copula variant/repeat artifact")
            if (
                parameters["posterior_draws"]
                != score_contract["numerics"]["posterior_draws"]
                or parameters["posterior_batch_size"]
                != score_contract["numerics"]["posterior_batch_size"]
                or manifest["master_seed"]
                != score_contract["numerics"]["master_seed"]
            ):
                raise ValueError("copula artifact numerics differ from the contract")
            rows_by_id = {row["sample_id"]: row for row in rows}
            if set(rows_by_id) != set(main_by_id):
                raise ValueError("copula artifact contains a different cohort")
            for sample_id in sample_ids:
                row = rows_by_id[sample_id]
                main = main_by_id[sample_id]
                if row["risk_dice"] != main["risk_dice"]:
                    raise ValueError(f"copula risk mismatch for {sample_id}")
                if row["image_index"] != main["image_index"]:
                    raise ValueError(f"copula image-index mismatch for {sample_id}")
            scores = np.asarray(
                [
                    rows_by_id[sample_id]["confidence_dice_spatial_copula"]
                    for sample_id in sample_ids
                ],
                dtype=float,
            )
            if not np.isfinite(scores).all():
                raise ValueError("copula artifact contains a non-finite score")
            repeat_scores[variant_id][repeat_index] = scores
            repeat_runtime[variant_id][repeat_index] = float(
                sum(rows_by_id[key]["score_runtime_seconds"] for key in sample_ids)
            )
            manifest_sha256[variant_id][repeat_index] = sha256_file(manifest_path)
            source_sha256.add(manifest["source_sha256"])
        if len(source_sha256) != 1:
            raise ValueError("copula repeats were produced by different source versions")

        baseline_scores = {
            "dice_m32": np.asarray(
                [row["confidence_dice_m32"] for row in main_rows], dtype=float
            ),
            "sdc": np.asarray(
                [row["confidence_sdc"] for row in main_rows], dtype=float
            ),
            "foreground_entropy": np.asarray(
                [row["confidence_foreground_entropy"] for row in main_rows],
                dtype=float,
            ),
        }
        method_results = {
            "dice_m32": _score_summary(baseline_scores["dice_m32"], risk, "Dice-M32"),
            "sdc": _score_summary(baseline_scores["sdc"], risk, "SDC"),
            "foreground_entropy": _score_summary(
                baseline_scores["foreground_entropy"], risk, "Foreground entropy"
            ),
        }
        diagnostics = {}
        aggregate_scores = {}
        for variant_id in variants:
            observed_repeats = repeat_scores[variant_id]
            if set(observed_repeats) != set(repeat_indices):
                raise ValueError(f"{variant_id} does not contain every declared repeat")
            matrix = np.stack(
                [observed_repeats[index] for index in repeat_indices], axis=1
            )
            aggregate = matrix.mean(axis=1)
            aggregate_scores[variant_id] = aggregate
            method_results[variant_id] = _score_summary(
                aggregate, risk, variants[variant_id]["id"]
            )
            repeat_aurcs = [
                summarize_aurc(matrix[:, column], risk).aurc
                for column in range(matrix.shape[1])
            ]
            diagnostics[variant_id] = {
                "repeat_aurcs": repeat_aurcs,
                "repeat_aurc_range": float(max(repeat_aurcs) - min(repeat_aurcs)),
                "mean_per_image_repeat_standard_deviation": float(
                    np.std(matrix, axis=1, ddof=0).mean()
                ),
                "total_score_runtime_seconds_by_repeat": repeat_runtime[variant_id],
                "manifest_sha256_by_repeat": manifest_sha256[variant_id],
            }

        bootstrap = paired_cluster_bootstrap_aurc_test(
            aggregate_scores[primary_variant],
            baseline_scores["dice_m32"],
            risk,
            cluster_ids=sample_ids,
            n_resamples=n_resamples,
            seed=_partition_seed(
                contract["reporting"]["bootstrap_seed"],
                dataset,
                condition,
                primary_variant,
                "dice_m32",
            ),
        )
        primary_differences.append(bootstrap.difference)
        conditions.append(
            {
                "dataset": dataset,
                "condition": condition,
                "num_images": len(sample_ids),
                "main_manifest_sha256": sha256_file(main_path),
                "copula_source_sha256": next(iter(source_sha256)),
                "methods": method_results,
                "diagnostics": diagnostics,
                "primary_comparison": {
                    "variant_id": primary_variant,
                    "reference": "dice_m32",
                    "difference_copula_minus_dice_m32": bootstrap.difference,
                    "bootstrap": asdict(bootstrap),
                },
            }
        )

    differences = np.asarray(primary_differences)
    gate = contract["primary_comparison"]["support_gate"]
    summary = {
        "primary_variant": primary_variant,
        "mean_difference_copula_minus_dice_m32": float(differences.mean()),
        "median_difference_copula_minus_dice_m32": float(np.median(differences)),
        "copula_wins": int(np.count_nonzero(differences < 0)),
        "dice_m32_wins": int(np.count_nonzero(differences > 0)),
        "ties": int(np.count_nonzero(differences == 0)),
    }
    summary["directional_gate_passed"] = bool(
        summary["copula_wins"] >= gate["required_condition_wins"]
        and summary["mean_difference_copula_minus_dice_m32"] < 0
    )
    return {
        "schema_version": 1,
        "analysis_id": contract["analysis_id"],
        "contract": {"path": str(contract_path), "sha256": contract_sha256},
        "score_contract": score_binding,
        "analysis": {
            "tie_policy": contract["reporting"]["tie_policy"],
            "bootstrap_resamples": n_resamples,
            "aurc_scale_for_display": contract["reporting"]["aurc_scale"],
            "selection_policy": "primary prespecified; all sensitivity variants reported",
            "repeat_policy": "independent jobs aggregated only after strict validation",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "conditions": conditions,
        "summary": summary,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "copula":
        result = analyze_spatial_copula(
            Path(args.contract),
            Path(args.main_root),
            Path(args.copula_root),
            bootstrap_resamples=args.bootstrap_resamples,
        )
        output = Path(args.output or Path(args.copula_root) / "analysis.json")
    elif args.mode == "partition":
        result = analyze_partitions(
            Path(args.contract),
            Path(args.main_root),
            Path(args.count_root),
            Path(args.partition_root),
            bootstrap_resamples=args.bootstrap_resamples,
        )
        output = (
            Path(args.output)
            if args.output is not None
            else Path(args.partition_root) / "analysis.json"
        )
    else:
        result = analyze(
            Path(args.contract),
            Path(args.main_root),
            Path(args.count_root),
            bootstrap_resamples=args.bootstrap_resamples,
        )
        output = Path(args.output or "outputs/dice_count_posterior_v1/analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"analysis output already exists: {output}")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
