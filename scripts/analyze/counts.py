"""Strict analysis of the predeclared Dice count-posterior experiment."""

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
    parser.add_argument(
        "--contract",
        default="configs/auxiliary/dice_count_result_analysis_v1.json",
    )
    parser.add_argument("--main-root", default="outputs/binary_midpoint_main_v2/assembled")
    parser.add_argument("--count-root", default="outputs/dice_count_posterior_v1")
    parser.add_argument(
        "--output", default="outputs/dice_count_posterior_v1/analysis.json"
    )
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
                    raise ValueError(
                        f"locked control mismatch for {main['sample_id']}/{main_field}"
                    )
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


def main(argv=None):
    args = parse_args(argv)
    result = analyze(
        Path(args.contract),
        Path(args.main_root),
        Path(args.count_root),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"analysis output already exists: {output}")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
