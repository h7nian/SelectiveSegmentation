"""Strict predeclared analysis of Ensemble-Q versus LevelSet-Q."""

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
from selectseg.artifacts import sha256_file
from selectseg.confidence import summarize_aurc


SCHEMA_VERSION = 1
MAIN_ARTIFACT_TYPE = "selectseg.binary_simulation_assembly"
BASELINE_ARTIFACT_TYPE = "selectseg.binary_ensemble_baselines"
RISKS = ("risk_dice", "risk_nhd", "risk_nhd95")
METHODS = (
    ("confidence_dice_m32", "LevelSet-Q Dice"),
    ("confidence_nhd_m32", "LevelSet-Q nHD"),
    ("confidence_nhd95_m32", "LevelSet-Q nHD95"),
    ("confidence_ensemble_q_dice", "Ensemble-Q Dice"),
    ("confidence_ensemble_q_nhd", "Ensemble-Q nHD"),
    ("confidence_ensemble_q_nhd95", "Ensemble-Q nHD95"),
    ("confidence_sdc", "SDC"),
    ("confidence_mean_max_probability", "Mean max probability"),
    ("confidence_negative_entropy", "Negative entropy"),
    ("confidence_threshold_iou_stability", "Threshold-IoU stability"),
    ("confidence_ensemble_all_iou", "All-member IoU"),
    ("confidence_ensemble_pairwise_dice", "Pairwise Dice"),
    ("confidence_ensemble_negative_mutual_information", "Negative mutual information"),
    ("confidence_ensemble_negative_probability_variance", "Negative probability variance"),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="configs/auxiliary/ensemble_posterior_analysis_v1.json",
    )
    parser.add_argument(
        "--main-root", default="outputs/binary_probability_ensemble_v1/assembled"
    )
    parser.add_argument(
        "--baseline-root",
        default="outputs/binary_probability_ensemble_v1/ensemble_baselines",
    )
    parser.add_argument(
        "--output", default="outputs/binary_probability_ensemble_v1/analysis.json"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _one_manifest(root: Path, dataset: str, condition: str) -> Path:
    candidates = sorted((root / dataset / condition).glob("*/manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one artifact for {dataset}/{condition} under {root}"
        )
    return candidates[0]


def _load_artifact(path: Path, artifact_type: str) -> tuple[dict, list[dict]]:
    manifest = _load_json(path)
    if manifest.get("artifact_type") != artifact_type:
        raise ValueError(f"{path} has unexpected artifact type")
    records_path = path.parent / "records.jsonl"
    expected_hash = manifest.get("records_sha256", manifest.get("jsonl_sha256"))
    if not isinstance(expected_hash, str) or sha256_file(records_path) != expected_hash:
        raise ValueError(f"{records_path} SHA-256 mismatch")
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    expected_rows = manifest.get("num_rows", manifest.get("num_images"))
    if len(rows) != expected_rows or not rows:
        raise ValueError(f"{records_path} has an unexpected row count")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) for value in sample_ids):
        raise ValueError(f"{records_path} contains an invalid sample ID")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{records_path} contains duplicate sample IDs")
    return manifest, rows


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
    return {
        "spearman_rho_with_safety": spearman,
        "kendall_tau_b_with_safety": kendall,
    }


def _seed(base: int, *parts: str) -> int:
    payload = "|".join([str(base), *parts]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _join_rows(main_rows: list[dict], baseline_rows: list[dict]) -> list[dict]:
    baseline = {row["sample_id"]: row for row in baseline_rows}
    if set(baseline) != {row["sample_id"] for row in main_rows}:
        raise ValueError("main and ensemble artifacts contain different cohorts")
    combined = []
    for main in main_rows:
        auxiliary = baseline[main["sample_id"]]
        for field in (*RISKS, "image_index"):
            if main[field] != auxiliary[field]:
                raise ValueError(
                    f"fixed-action field mismatch for {main['sample_id']}/{field}"
                )
        overlap = set(main) & set(auxiliary) - {"schema_version", "sample_id"}
        if overlap - {*RISKS, "image_index"}:
            raise ValueError(f"unexpected overlapping fields: {sorted(overlap)}")
        combined.append({**main, **{k: v for k, v in auxiliary.items() if k not in main}})
    return combined


def analyze(
    contract_path: Path,
    main_root: Path,
    baseline_root: Path,
    *,
    bootstrap_resamples: int | None = None,
    seed: int = 20260721,
) -> dict:
    contract = _load_json(contract_path)
    if contract.get("status") != "predeclared-before-reading-ensemble-baseline-outputs":
        raise ValueError("ensemble analysis contract has an unexpected status")
    expected = [
        (dataset, model)
        for dataset in contract["conditions"]["datasets"]
        for model in contract["conditions"]["models"]
    ]
    if len(expected) != contract["conditions"]["count"] or len(expected) != 10:
        raise ValueError("ensemble analysis contract must identify ten conditions")
    declared_primary = {
        comparison["risk"]: (comparison["left"], comparison["right"])
        for comparison in contract["primary_comparisons"]
    }
    if set(declared_primary) != {"dice", "nhd", "nhd95"}:
        raise ValueError("contract must declare one primary comparison per risk")
    n_resamples = (
        contract["reporting"]["bootstrap_resamples"]
        if bootstrap_resamples is None
        else bootstrap_resamples
    )
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")

    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": contract["analysis_id"],
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "analysis": {
            "aurc_scale_for_display": contract["reporting"]["aurc_scale"],
            "tie_policy": contract["reporting"]["tie_policy"],
            "bootstrap_resamples": n_resamples,
            "seed": seed,
        },
        "conditions": [],
        "summary": {},
    }
    differences = {risk: [] for risk in RISKS}
    for dataset, condition in sorted(expected):
        main_path = _one_manifest(main_root, dataset, condition)
        baseline_path = _one_manifest(baseline_root, dataset, condition)
        main_manifest, main_rows = _load_artifact(main_path, MAIN_ARTIFACT_TYPE)
        baseline_manifest, baseline_rows = _load_artifact(
            baseline_path, BASELINE_ARTIFACT_TYPE
        )
        if (main_manifest["dataset"], main_manifest["condition"]) != (dataset, condition):
            raise ValueError("main manifest identity mismatch")
        if (baseline_manifest["dataset"], baseline_manifest["condition"]) != (
            dataset,
            condition,
        ):
            raise ValueError("baseline manifest identity mismatch")
        rows = _join_rows(main_rows, baseline_rows)
        available = set(rows[0])
        required = set(RISKS) | {field for field, _ in METHODS}
        if not required.issubset(available):
            raise ValueError(f"joined rows lack fields: {sorted(required - available)}")
        condition_result = {
            "dataset": dataset,
            "condition": condition,
            "num_images": len(rows),
            "main_manifest_sha256": sha256_file(main_path),
            "baseline_manifest_sha256": sha256_file(baseline_path),
            "risks": {},
            "primary_comparisons": {},
        }
        sample_ids = [row["sample_id"] for row in rows]
        for risk_field in RISKS:
            risks = np.asarray([row[risk_field] for row in rows], dtype=float)
            method_results = {}
            for score_field, label in METHODS:
                scores = np.asarray([row[score_field] for row in rows], dtype=float)
                method_results[score_field] = {
                    "label": label,
                    **asdict(summarize_aurc(scores, risks)),
                    **_rank_metrics(scores, risks),
                }
            condition_result["risks"][risk_field] = {"methods": method_results}

            short_risk = risk_field.removeprefix("risk_")
            left, right = declared_primary[short_risk]
            left_scores = np.asarray([row[left] for row in rows], dtype=float)
            right_scores = np.asarray([row[right] for row in rows], dtype=float)
            bootstrap = paired_cluster_bootstrap_aurc_test(
                left_scores,
                right_scores,
                risks,
                cluster_ids=sample_ids,
                n_resamples=n_resamples,
                seed=_seed(seed, dataset, condition, risk_field),
            )
            comparison = {
                "left": left,
                "right": right,
                "difference_left_minus_right": bootstrap.difference,
                "bootstrap": asdict(bootstrap),
            }
            condition_result["primary_comparisons"][risk_field] = comparison
            differences[risk_field].append(bootstrap.difference)
        result["conditions"].append(condition_result)

    for risk, values in differences.items():
        array = np.asarray(values)
        level_set_wins = int(np.count_nonzero(array > 0))
        ensemble_wins = int(np.count_nonzero(array < 0))
        result["summary"][risk] = {
            "mean_difference_ensemble_minus_level_set": float(array.mean()),
            "median_difference_ensemble_minus_level_set": float(np.median(array)),
            "ensemble_q_wins": ensemble_wins,
            "level_set_q_wins": level_set_wins,
            "ties": int(np.count_nonzero(array == 0)),
        }
    dice_gate = contract["interpretation_gates"]["ensemble_supports_dice_expressivity_hypothesis"]
    result["summary"]["dice_gate_passed"] = bool(
        result["summary"]["risk_dice"]["ensemble_q_wins"]
        >= dice_gate["required_condition_wins"]
        and result["summary"]["risk_dice"]["mean_difference_ensemble_minus_level_set"] < 0
    )
    nhd_gate = contract["interpretation_gates"]["level_set_remains_preferred_for_nhd"]
    result["summary"]["nhd_level_set_gate_passed"] = bool(
        result["summary"]["risk_nhd"]["level_set_q_wins"]
        >= nhd_gate["required_condition_wins"]
    )
    return result


def main(argv=None):
    args = parse_args(argv)
    result = analyze(
        Path(args.contract),
        Path(args.main_root),
        Path(args.baseline_root),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
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
