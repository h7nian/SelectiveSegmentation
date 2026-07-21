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
METHODS = (
    ("confidence_dice_action_two_block_m32", "Dice two-block"),
    ("confidence_dice_m32", "Dice-M32"),
    ("confidence_dice_exact", "Dice-Exact"),
    ("confidence_sdc", "SDC"),
    ("confidence_foreground_entropy", "Foreground entropy"),
)
CONTROL_ABSOLUTE_TOLERANCE = 4 * np.finfo(np.float64).eps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("count", "partition"), default="count")
    parser.add_argument(
        "--contract",
        default="configs/auxiliary/dice_count_result_analysis_v1.json",
    )
    parser.add_argument("--main-root", default="outputs/binary_midpoint_main_v2/assembled")
    parser.add_argument("--count-root", default="outputs/dice_count_posterior_v1")
    parser.add_argument("--partition-root", default="outputs/dice_partition_ladder_v1")
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


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "partition":
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
