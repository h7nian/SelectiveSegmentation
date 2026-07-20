"""Strictly aggregate known-posterior synthetic cell artifacts.

The analyzer derives every expected cell directory from the immutable lock.
It accepts the 12-cell pilot or the complete 360-cell union (12 pilot plus 348
main artifacts), rejects missing/duplicate/stale cells, and never treats Monte
Carlo error as posterior misspecification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from selectseg.synthetic_posterior import (
    ARTIFACT_TYPE,
    LOSSES,
    MANIFEST_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    _canonical_json,
    _load_json,
    _resolve_from_repo,
    cell_seeds,
    load_synthetic_lock,
    pilot_cells,
    selected_cells,
)
from selectseg.threshold_estimators import sha256_file


ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_LOCK = "configs/auxiliary/synthetic_posterior-v1.lock.json"
_MANIFEST_FIELDS = frozenset(
    {
        "manifest_schema_version",
        "artifact_type",
        "artifact_id",
        "campaign_id",
        "created_utc",
        "phase",
        "cell",
        "seeds",
        "lock",
        "spec",
        "code_sources",
        "summary",
        "runtime_seconds",
        "environment",
        "command",
        "storage_policy",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "summary_schema_version",
        "cell",
        "seeds",
        "cohort",
        "posterior_discrepancy",
        "losses",
        "monte_carlo_note",
    }
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument("--mode", choices=("pilot", "complete"), default="pilot")
    parser.add_argument(
        "--output",
        default="outputs/synthetic_posterior_analysis/analysis.json",
    )
    return parser.parse_args(argv)


def _cell_dict(cell):
    return {
        "coupling": cell.coupling,
        "sharpness": cell.sharpness,
        "morphology": cell.morphology,
        "replicate": cell.replicate,
    }


def _cell_root(binding, cell, phase):
    path_key = "pilot_output_root" if phase == "pilot" else "main_output_root"
    return (
        _resolve_from_repo(binding["path"], binding["spec"]["paths"][path_key])
        / cell.coupling
        / cell.sharpness
        / cell.morphology
        / f"replicate-{cell.replicate:02d}"
    )


def _phase_for_cell(binding, cell, mode):
    if mode == "pilot":
        return "pilot"
    return "pilot" if cell in set(pilot_cells(binding["spec"])) else "full"


def _load_one_cell(binding, cell, phase):
    cell_root = _cell_root(binding, cell, phase)
    if not cell_root.is_dir():
        raise FileNotFoundError(f"missing synthetic cell directory: {cell_root}")
    candidates = sorted(
        path
        for path in cell_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one published artifact in {cell_root}, found {len(candidates)}"
        )
    artifact = candidates[0]
    manifest_path = artifact / "manifest.json"
    _, manifest = _load_json(manifest_path, name="synthetic manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError(f"manifest has unexpected fields: {manifest_path}")
    if (
        manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_type") != ARTIFACT_TYPE
    ):
        raise ValueError(f"manifest schema/type mismatch: {manifest_path}")
    if manifest.get("artifact_id") != artifact.name:
        raise ValueError(
            f"artifact directory and manifest ID disagree: {manifest_path}"
        )
    if manifest.get("campaign_id") != binding["spec"]["campaign_id"]:
        raise ValueError(f"campaign mismatch: {manifest_path}")
    if manifest.get("phase") != phase or manifest.get("cell") != _cell_dict(cell):
        raise ValueError(f"phase/cell mismatch: {manifest_path}")
    if manifest.get("seeds") != cell_seeds(binding["spec"], cell):
        raise ValueError(f"seed mismatch: {manifest_path}")
    if manifest.get("lock", {}).get("sha256") != binding["sha256"]:
        raise ValueError(f"lock hash mismatch: {manifest_path}")
    if manifest.get("spec", {}).get("sha256") != binding["spec_sha256"]:
        raise ValueError(f"spec hash mismatch: {manifest_path}")
    observed_sources = [
        entry.get("sha256") for entry in manifest.get("code_sources", [])
    ]
    if observed_sources != [sha for _, sha in binding["code_sources"]]:
        raise ValueError(f"source binding mismatch: {manifest_path}")
    summary_binding = manifest.get("summary")
    if not isinstance(summary_binding, dict) or set(summary_binding) != {
        "path",
        "sha256",
    }:
        raise ValueError(f"invalid summary binding: {manifest_path}")
    if summary_binding["path"] != "summary.json":
        raise ValueError(f"summary must be artifact-local: {manifest_path}")
    summary_path = artifact / "summary.json"
    if sha256_file(summary_path) != summary_binding["sha256"]:
        raise ValueError(f"summary hash mismatch: {summary_path}")
    _, summary = _load_json(summary_path, name="synthetic summary")
    if (
        set(summary) != _SUMMARY_FIELDS
        or summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION
    ):
        raise ValueError(f"summary schema mismatch: {summary_path}")
    if (
        summary.get("cell") != _cell_dict(cell)
        or summary.get("seeds") != manifest["seeds"]
    ):
        raise ValueError(f"summary identity mismatch: {summary_path}")
    if set(summary.get("losses", {})) != set(LOSSES):
        raise ValueError(f"summary loss set mismatch: {summary_path}")
    runtime = manifest.get("runtime_seconds")
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)):
        raise TypeError(f"runtime must be numeric: {manifest_path}")
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"runtime must be finite and nonnegative: {manifest_path}")
    return {
        "cell": cell,
        "phase": phase,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "runtime_seconds": float(runtime),
        "summary": summary,
    }


def _mean(values):
    finite = [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]
    return None if not finite else float(np.mean(finite))


def _aggregate_group(records):
    output = {
        "num_cells": len(records),
        "runtime_seconds": {
            "mean": _mean(record["runtime_seconds"] for record in records),
            "maximum": float(max(record["runtime_seconds"] for record in records)),
        },
        "posterior_discrepancy": {
            "mean_exact_tv": _mean(
                record["summary"]["posterior_discrepancy"]["total_variation_exact"][
                    "mean"
                ]
                for record in records
            ),
            "mean_empty_event_tv_lower_bound": _mean(
                record["summary"]["posterior_discrepancy"][
                    "empty_event_tv_lower_bound_exact"
                ]["mean"]
                for record in records
            ),
            "mean_paired_jaccard_transport_upper_bound": _mean(
                record["summary"]["posterior_discrepancy"][
                    "paired_jaccard_transport_cost_upper_bound"
                ]["mean"]
                for record in records
            ),
            "mean_paired_nhd_transport_upper_bound": _mean(
                record["summary"]["posterior_discrepancy"][
                    "paired_normalized_hd_transport_cost_upper_bound"
                ]["mean"]
                for record in records
            ),
        },
        "losses": {},
    }
    for loss in LOSSES:
        estimators = {}
        names = (
            ("m2", "m8", "m32", "m128", "exact")
            if loss == "dice"
            else (
                "m2",
                "m8",
                "m32",
                "m128",
            )
        )
        for name in names:
            entries = [
                record["summary"]["losses"][loss]["estimators"][name]
                for record in records
            ]
            estimators[name] = {
                "mean_absolute_score_error": _mean(
                    entry["score_error"]["mean"] for entry in entries
                ),
                "mean_signed_score_error": _mean(
                    entry["score_error"]["signed_bias"] for entry in entries
                ),
                "mean_spearman_risk_ranking": _mean(
                    entry["spearman_risk_ranking"] for entry in entries
                ),
                "mean_kendall_tau_b_risk_ranking": _mean(
                    entry["kendall_tau_b_risk_ranking"] for entry in entries
                ),
                "mean_aurc_regret": _mean(entry["aurc_regret"] for entry in entries),
                "mean_aurc_regret_mc_se": _mean(
                    entry["aurc_regret_mc_se"] for entry in entries
                ),
            }
        output["losses"][loss] = {
            "estimators": estimators,
            "mean_loss_pushforward_w1_empirical": _mean(
                record["summary"]["losses"][loss]["loss_pushforward_w1_empirical"][
                    "mean"
                ]
                for record in records
            ),
            "mean_true_risk_mc_se": _mean(
                record["summary"]["losses"][loss]["cell_mean_true_risk_mc_se"]
                for record in records
            ),
        }
    return output


def _pilot_gate(records):
    shared = [
        record for record in records if record["cell"].coupling == "shared_threshold"
    ]
    reasons = []
    for record in shared:
        summary = record["summary"]
        discrepancy = summary["posterior_discrepancy"]
        exact_zero = (
            discrepancy["total_variation_exact"]["maximum"] == 0.0
            and discrepancy["paired_jaccard_transport_cost_upper_bound"]["maximum"]
            == 0.0
            and discrepancy["paired_normalized_hd_transport_cost_upper_bound"][
                "maximum"
            ]
            == 0.0
        )
        if not exact_zero:
            reasons.append(
                f"{record['cell'].slug}: well-specified P/Q discrepancy was nonzero"
            )
        for loss in LOSSES:
            values = summary["losses"][loss]
            if values["cell_mean_qmc_minus_true"] != 0.0:
                reasons.append(
                    f"{record['cell'].slug}/{loss}: paired P/Q MC risks differed"
                )
        exact = summary["losses"]["dice"]["estimators"]["exact"]["score_error"]
        se = summary["losses"]["dice"]["cell_mean_true_risk_mc_se"]
        if abs(exact["signed_bias"]) > 3.0 * max(se, np.finfo(float).eps):
            reasons.append(f"{record['cell'].slug}: exact Dice exceeded 3 MC SE")
    maximum_runtime = max(record["runtime_seconds"] for record in records)
    if maximum_runtime > 3 * 60 * 60:
        reasons.append(
            "maximum pilot runtime lacks 25% headroom under the four-hour cap"
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "criteria": {
            "shared_threshold_paired_mc_identity": True,
            "shared_threshold_exact_dice_within_three_mc_se": True,
            "maximum_runtime_seconds": 10800,
        },
        "observed_maximum_runtime_seconds": float(maximum_runtime),
    }


def analyze(lock, *, mode):
    binding = load_synthetic_lock(lock)
    expected = selected_cells(
        binding["spec"], "pilot" if mode == "pilot" else "complete"
    )
    records = []
    for cell in expected:
        phase = _phase_for_cell(binding, cell, mode)
        records.append(_load_one_cell(binding, cell, phase))
    expected_count = 12 if mode == "pilot" else 360
    if len(records) != expected_count:
        raise RuntimeError(f"expected {expected_count} validated records")

    by_full_group = defaultdict(list)
    by_headline_group = defaultdict(list)
    by_coupling = defaultdict(list)
    for record in records:
        cell = record["cell"]
        by_full_group[(cell.coupling, cell.sharpness, cell.morphology)].append(record)
        by_headline_group[(cell.coupling, cell.sharpness)].append(record)
        by_coupling[cell.coupling].append(record)

    sources = [
        {
            "path": str(record["manifest_path"]),
            "sha256": record["manifest_sha256"],
        }
        for record in records
    ]
    analysis_identity = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "lock_sha256": binding["sha256"],
        "mode": mode,
        "source_manifest_sha256": [source["sha256"] for source in sources],
    }
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": hashlib.sha256(
            _canonical_json(analysis_identity).encode()
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign_id": binding["spec"]["campaign_id"],
        "mode": mode,
        "lock": {"path": str(binding["path"]), "sha256": binding["sha256"]},
        "num_cells": len(records),
        "source_manifests": sources,
        "pilot_gate": _pilot_gate(records) if mode == "pilot" else None,
        "groups": [
            {
                "coupling": key[0],
                "sharpness": key[1],
                "morphology": key[2],
                **_aggregate_group(group),
            }
            for key, group in sorted(by_full_group.items())
        ],
        "headline_groups": [
            {
                "coupling": key[0],
                "sharpness": key[1],
                **_aggregate_group(group),
            }
            for key, group in sorted(by_headline_group.items())
        ],
        "coupling_summaries": [
            {"coupling": key, **_aggregate_group(group)}
            for key, group in sorted(by_coupling.items())
        ],
        "interpretation": {
            "target": "misspecification of Q_p under known mask-posterior couplings",
            "monte_carlo": (
                "True conditional risks and paired transport costs are Monte Carlo "
                "estimates with separately reported posterior-integration SE."
            ),
            "total_variation": "exact for all four constructed finite posterior laws",
            "wasserstein": (
                "Jaccard/full-HD paired costs upper-bound optimal mask transport; "
                "scalar empirical pushforward W1 applies to all losses."
            ),
            "hd95": "No HD95 mask-Wasserstein corollary is asserted.",
        },
    }


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {output}")
    analysis = analyze(Path(args.lock), mode=args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
