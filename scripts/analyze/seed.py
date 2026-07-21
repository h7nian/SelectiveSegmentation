"""Strict descriptive analysis of the three target-model training seeds.

Seed 0 is read from the locked primary assemblies; seeds 1 and 2 are read from
the two canonical-compatible campaigns bound by the seed downstream lock.
Every assembly is derived from its exact content-addressed common/M shards and
validated with the canonical analyzer loader.  The three checkpoints are the
replicates: this script reports values, mean, range, and sample standard
deviation, and deliberately performs no image-pooled or seed-level test.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from scripts.analyze.main import CONTRASTS, METHODS, RISKS, load_condition
from scripts.assemble import (
    load_campaign_lock as load_assembly_lock,
    prepare_assembly,
)
from scripts.submit.main import (
    _expected_score_manifests,
    _project_path,
    load_campaign_lock,
)
from selectseg.confidence import summarize_aurc
from selectseg.seed.downstream import load_downstream_lock
from selectseg.seed.extension import _atomic_write_new, _load_json, _sha256


ANALYSIS_SCHEMA_VERSION = 1
TARGET_CONDITIONS = ("clipseg-target", "deeplabv3-target")
TARGET_MODELS = {
    "clipseg-target": "clipseg",
    "deeplabv3-target": "deeplabv3",
}
TARGET_DATASETS = ("pet", "kvasir", "fives", "isic", "tn3k")
COHORT_JOIN_FIELDS = (
    "sample_id",
    "image_id",
    "image_index",
    "class_index",
    "class_name",
    "height",
    "width",
    "image_diagonal",
    "truth_foreground_fraction",
)
COHORT_FLOAT_FIELDS = frozenset({"image_diagonal", "truth_foreground_fraction"})
COHORT_REL_TOL = 2e-12
COHORT_ABS_TOL = 2e-12


def _analysis_source_sha256():
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "scripts" / "analyze" / "main.py",
        root / "scripts" / "assemble.py",
        root / "scripts" / "submit" / "main.py",
        root / "selectseg" / "confidence.py",
        root / "selectseg" / "seed" / "downstream.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-lock", required=True)
    parser.add_argument("--expected-downstream-lock-sha256", required=True)
    parser.add_argument("--canonical-analysis", required=True)
    parser.add_argument("--expected-canonical-analysis-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _strict_digest(value, *, location):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value.lower()


def _regular_file(path, expected_sha, *, name):
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} must be a regular non-symlink file: {source}")
    expected = _strict_digest(expected_sha, location=f"expected {name} SHA-256")
    observed = _sha256(source)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return source.resolve(), observed


def _seed_assembly_conditions(downstream_binding):
    result = {}
    for campaign_binding in downstream_binding["campaigns"]:
        seed = campaign_binding["training_seed"]
        config = campaign_binding["config"]
        lock_path = campaign_binding["campaign_lock_path"]
        lock_path, lock_sha, lock = load_campaign_lock(lock_path, config=config)
        assembly_lock = load_assembly_lock(lock_path)
        conditions = {}
        for artifact in lock["artifacts"]:
            common, simulations = _expected_score_manifests(
                config, lock_path, lock_sha, lock, artifact
            )
            missing = [path for path in (common, *simulations) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"seed-{seed} scoring outputs are incomplete: {missing}"
                )
            dataset, condition, run_id, expected_records, expected_manifest = (
                prepare_assembly(assembly_lock, common, simulations)
            )
            assembly_root = _project_path(
                config.path, lock["paths"]["assembly_output_root"]
            )
            records_path = (
                assembly_root / dataset / condition / run_id / "records.jsonl"
            )
            manifest_path = records_path.with_name("manifest.json")
            if not records_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(
                    f"seed-{seed} assembly is incomplete: {records_path.parent}"
                )
            if records_path.read_bytes() != expected_records:
                raise ValueError(
                    "assembled seed records differ from their strict shards"
                )
            if manifest_path.read_bytes() != expected_manifest:
                raise ValueError(
                    "assembled seed manifest differs from its strict shards"
                )
            loaded = load_condition(records_path)
            key = (dataset, condition)
            if key in conditions:
                raise ValueError(f"duplicate seed-{seed} assembly condition {key}")
            conditions[key] = loaded
        if len(conditions) != 10:
            raise ValueError(f"seed-{seed} campaign must contain ten assemblies")
        result[seed] = conditions
    if set(result) != {1, 2}:
        raise ValueError("seed downstream lock must supply seed 1 and seed 2")
    return result


def _canonical_seed0_conditions(
    downstream_binding,
    *,
    canonical_analysis,
    expected_canonical_analysis_sha256,
):
    analysis_path, analysis_sha = _regular_file(
        canonical_analysis,
        expected_canonical_analysis_sha256,
        name="canonical analysis",
    )
    analysis = _load_json(analysis_path)
    if analysis.get("schema_version") != 2:
        raise ValueError("canonical analysis has an unsupported schema")
    provenance = analysis.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("binding") != "campaign-lock":
        raise ValueError("canonical analysis is not campaign-lock bound")
    canonical_binding = downstream_binding["binding"]["lock"]["canonical_campaign_lock"]
    if provenance.get("campaign_lock", {}).get("sha256") != canonical_binding["sha256"]:
        raise ValueError("canonical analysis names a different primary campaign lock")
    main_lock_path = Path(canonical_binding["path"])
    main_lock_path, main_lock_sha, main_lock = load_campaign_lock(main_lock_path)
    if main_lock_sha != canonical_binding["sha256"]:
        raise ValueError("primary campaign-lock bytes changed")
    target_artifacts = {
        (row["dataset"], row["condition"]): row
        for row in main_lock["artifacts"]
        if row["condition"] in TARGET_CONDITIONS
    }
    if len(target_artifacts) != 10:
        raise ValueError("primary campaign must contain ten target conditions")

    analysis_conditions = {
        (row["dataset"], row["condition"]): row
        for row in analysis.get("conditions", [])
        if row.get("condition") in TARGET_CONDITIONS
    }
    input_bindings = {
        (row["dataset"], row["condition"]): row
        for row in provenance.get("inputs", [])
        if row.get("condition") in TARGET_CONDITIONS
    }
    if set(analysis_conditions) != set(target_artifacts) or set(input_bindings) != set(
        target_artifacts
    ):
        raise ValueError("canonical analysis target-condition set is incomplete")
    assembly_root = _project_path(
        main_lock_path, main_lock["paths"]["assembly_output_root"]
    )
    result = {}
    for key, input_binding in input_bindings.items():
        records_path = assembly_root / input_binding["logical_id"] / "records.jsonl"
        manifest_path = records_path.with_name("manifest.json")
        if _sha256(records_path) != input_binding["records_sha256"]:
            raise ValueError(f"canonical seed-0 records changed for {key}")
        if _sha256(manifest_path) != input_binding["manifest_sha256"]:
            raise ValueError(f"canonical seed-0 manifest changed for {key}")
        loaded = load_condition(records_path)
        artifact = target_artifacts[key]
        if (
            loaded.manifest["sample_id_sha256"] != artifact["sample_id_sha256"]
            or loaded.manifest["num_images"] != artifact["num_samples"]
        ):
            raise ValueError(f"canonical seed-0 cohort differs from its lock for {key}")
        result[key] = loaded

    # Recompute every paper-facing point estimate and demand exact agreement
    # with the frozen primary analysis before it can anchor seed robustness.
    summaries = {key: _summarize_condition(run) for key, run in result.items()}
    for key, summary in summaries.items():
        frozen = analysis_conditions[key]
        for risk, methods in summary["raw_aurc"].items():
            for method, value in methods.items():
                expected = frozen["risks"][risk]["methods"][method]["aurc"]
                if value != expected:
                    raise ValueError(
                        f"canonical seed-0 AURC no longer reproduces for {key}/{risk}/{method}"
                    )
        for name, value in summary["contrasts"].items():
            expected = frozen["comparisons"][name]["difference_left_minus_right"]
            if value != expected:
                raise ValueError(
                    f"canonical seed-0 contrast no longer reproduces for {key}/{name}"
                )
    return result, {
        "path": analysis_path.as_posix(),
        "sha256": analysis_sha,
        "campaign_lock_path": main_lock_path.as_posix(),
        "campaign_lock_sha256": main_lock_sha,
    }


def _strict_cohort_join(reference, candidate, *, context):
    if reference.manifest["sample_id_sha256"] != candidate.manifest["sample_id_sha256"]:
        raise ValueError(f"{context}: sample-order digest differs")
    if len(reference.rows) != len(candidate.rows):
        raise ValueError(f"{context}: cohort size differs")
    for left, right in zip(reference.rows, candidate.rows, strict=True):
        for field in COHORT_JOIN_FIELDS:
            if field in COHORT_FLOAT_FIELDS:
                agrees = math.isclose(
                    float(left[field]),
                    float(right[field]),
                    rel_tol=COHORT_REL_TOL,
                    abs_tol=COHORT_ABS_TOL,
                )
            else:
                agrees = left[field] == right[field]
            if not agrees:
                raise ValueError(f"{context}: cohort field {field!r} differs")


def validate_analysis_inputs(
    downstream_binding,
    *,
    canonical_analysis,
    expected_canonical_analysis_sha256,
):
    """Validate all 30 assemblies and exact held-out cohort joins."""

    seed0, canonical_provenance = _canonical_seed0_conditions(
        downstream_binding,
        canonical_analysis=canonical_analysis,
        expected_canonical_analysis_sha256=expected_canonical_analysis_sha256,
    )
    extension = _seed_assembly_conditions(downstream_binding)
    expected_keys = set(seed0)
    for seed in (1, 2):
        if set(extension[seed]) != expected_keys:
            raise ValueError(f"seed-{seed} condition set differs from seed 0")
    for key in sorted(expected_keys):
        for seed in (1, 2):
            _strict_cohort_join(
                seed0[key], extension[seed][key], context=f"{key}, seed 0 vs {seed}"
            )
    return {0: seed0, **extension}, canonical_provenance


def _summarize_condition(run):
    raw_aurc = {}
    for risk, _ in RISKS:
        risks = [row[risk] for row in run.rows]
        raw_aurc[risk] = {
            method: asdict(summarize_aurc([row[method] for row in run.rows], risks))[
                "aurc"
            ]
            for method, _ in METHODS
        }
    contrasts = {
        contrast.name: (
            raw_aurc[contrast.risk][contrast.left]
            - raw_aurc[contrast.risk][contrast.right]
        )
        for contrast in CONTRASTS
    }
    return {"raw_aurc": raw_aurc, "contrasts": contrasts}


def _direction(value):
    if not math.isfinite(value):
        raise ValueError("seed summary contains a non-finite value")
    return 1 if value > 0 else -1 if value < 0 else 0


def _finite_number(value, *, location, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    if lower is not None and result < lower:
        raise ValueError(f"{location} must be at least {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{location} must be at most {upper}")
    return result


def _same_float(observed, expected, *, location):
    value = _finite_number(observed, location=location)
    if not math.isclose(value, expected, rel_tol=1e-15, abs_tol=1e-15):
        raise ValueError(f"{location} is inconsistent with the three seed values")


def _validate_three_seed_summary(summary, *, location, contrast):
    base_fields = {
        "values",
        "mean",
        "minimum",
        "maximum",
        "range",
        "sample_standard_deviation",
    }
    contrast_fields = {
        "directions",
        "majority_direction",
        "seed0_is_majority_direction",
        "direction_reversal",
    }
    expected_fields = base_fields | (contrast_fields if contrast else set())
    if not isinstance(summary, dict) or set(summary) != expected_fields:
        raise ValueError(f"{location} has an unexpected summary schema")
    values = summary["values"]
    if not isinstance(values, dict) or set(values) != {"0", "1", "2"}:
        raise ValueError(f"{location}.values must contain exactly seeds 0, 1, and 2")
    lower, upper = (-1.0, 1.0) if contrast else (0.0, 1.0)
    ordered = [
        _finite_number(
            values[str(seed)],
            location=f"{location}.values.{seed}",
            lower=lower,
            upper=upper,
        )
        for seed in (0, 1, 2)
    ]
    expected_statistics = {
        "mean": float(statistics.fmean(ordered)),
        "minimum": min(ordered),
        "maximum": max(ordered),
        "range": max(ordered) - min(ordered),
        "sample_standard_deviation": float(statistics.stdev(ordered)),
    }
    for field, expected in expected_statistics.items():
        _same_float(summary[field], expected, location=f"{location}.{field}")
    if contrast:
        directions = {str(seed): _direction(ordered[seed]) for seed in (0, 1, 2)}
        if summary["directions"] != directions:
            raise ValueError(f"{location}.directions are inconsistent")
        counts = {
            direction: tuple(directions.values()).count(direction)
            for direction in (-1, 0, 1)
        }
        majority = [direction for direction, count in counts.items() if count >= 2]
        expected_majority = majority[0] if len(majority) == 1 else None
        if summary["majority_direction"] != expected_majority:
            raise ValueError(f"{location}.majority_direction is inconsistent")
        expected_seed0_majority = (
            expected_majority is not None and directions["0"] == expected_majority
        )
        if summary["seed0_is_majority_direction"] is not expected_seed0_majority:
            raise ValueError(f"{location}.seed0_is_majority_direction is inconsistent")
        expected_reversal = -1 in directions.values() and 1 in directions.values()
        if summary["direction_reversal"] is not expected_reversal:
            raise ValueError(f"{location}.direction_reversal is inconsistent")
    return values


def _validate_path_digest(binding, *, location):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{location} must contain exactly path and sha256")
    if not isinstance(binding["path"], str) or not binding["path"]:
        raise ValueError(f"{location}.path must be a non-empty string")
    _strict_digest(binding["sha256"], location=f"{location}.sha256")


def validate_analysis_document(result, *, require_current_source=True):
    """Recompute every stored seed statistic, contrast, and Gate C decision."""

    if not isinstance(result, dict) or set(result) != {
        "schema_version",
        "analysis",
        "provenance",
        "cells",
        "gate_c",
    }:
        raise ValueError("seed analysis has an unexpected top-level schema")
    if result["schema_version"] != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported seed-analysis schema")
    expected_analysis = {
        "estimand": "descriptive target-model training-seed variation over seeds 0,1,2",
        "replication_unit": "one independently trained checkpoint",
        "inference": "none; no image pooling and no seed-level hypothesis test",
        "statistics": "three values, mean, range, and sample standard deviation",
        "aurc_scale": "raw [0,1]; renderers may display 100 x AURC",
        "contrast_definition": "AURC(left score) - AURC(right score)",
        "contrast_definitions": [asdict(contrast) for contrast in CONTRASTS],
        "cohort_join_fields": list(COHORT_JOIN_FIELDS),
    }
    if result["analysis"] != expected_analysis:
        raise ValueError("seed analysis metadata differs from the locked estimand")

    provenance = result["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "downstream_lock",
        "canonical_seed0",
        "analysis_source_sha256",
    }:
        raise ValueError("seed analysis has an unexpected provenance schema")
    _validate_path_digest(
        provenance["downstream_lock"], location="provenance.downstream_lock"
    )
    canonical = provenance["canonical_seed0"]
    if not isinstance(canonical, dict) or set(canonical) != {
        "path",
        "sha256",
        "campaign_lock_path",
        "campaign_lock_sha256",
    }:
        raise ValueError("canonical seed-0 provenance has an unexpected schema")
    for path_field, digest_field in (
        ("path", "sha256"),
        ("campaign_lock_path", "campaign_lock_sha256"),
    ):
        if not isinstance(canonical[path_field], str) or not canonical[path_field]:
            raise ValueError(f"canonical_seed0.{path_field} must be non-empty")
        _strict_digest(
            canonical[digest_field], location=f"canonical_seed0.{digest_field}"
        )
    source_sha = _strict_digest(
        provenance["analysis_source_sha256"],
        location="provenance.analysis_source_sha256",
    )
    if require_current_source and source_sha != _analysis_source_sha256():
        raise ValueError(
            "seed analysis was produced by different analysis source bytes"
        )

    cells = result["cells"]
    if not isinstance(cells, list) or len(cells) != 10:
        raise ValueError("seed analysis must contain exactly ten target cells")
    by_key = {}
    source_paths = set()
    method_names = {method for method, _ in METHODS}
    risk_names = {risk for risk, _ in RISKS}
    contrast_by_name = {contrast.name: contrast for contrast in CONTRASTS}
    for index, cell in enumerate(cells):
        location = f"cells[{index}]"
        if not isinstance(cell, dict) or set(cell) != {
            "dataset",
            "condition",
            "model",
            "num_images_per_seed",
            "sources",
            "summary",
        }:
            raise ValueError(f"{location} has an unexpected schema")
        key = (cell["dataset"], cell["condition"])
        if key in by_key:
            raise ValueError(f"duplicate seed-analysis cell {key}")
        if key[0] not in TARGET_DATASETS or key[1] not in TARGET_CONDITIONS:
            raise ValueError(f"unexpected seed-analysis cell {key}")
        if cell["model"] != TARGET_MODELS[key[1]]:
            raise ValueError(f"{location}.model differs from its target condition")
        if (
            isinstance(cell["num_images_per_seed"], bool)
            or not isinstance(cell["num_images_per_seed"], int)
            or cell["num_images_per_seed"] <= 0
        ):
            raise ValueError(f"{location}.num_images_per_seed must be positive")
        sources = cell["sources"]
        if not isinstance(sources, dict) or set(sources) != {"0", "1", "2"}:
            raise ValueError(f"{location}.sources must contain exactly three seeds")
        for seed, source in sources.items():
            if not isinstance(source, dict) or set(source) != {
                "records",
                "records_sha256",
                "manifest",
                "manifest_sha256",
            }:
                raise ValueError(f"{location}.sources.{seed} has an unexpected schema")
            for path_field, digest_field in (
                ("records", "records_sha256"),
                ("manifest", "manifest_sha256"),
            ):
                path = source[path_field]
                if not isinstance(path, str) or not path:
                    raise ValueError(
                        f"{location}.sources.{seed}.{path_field} must be non-empty"
                    )
                if path in source_paths:
                    raise ValueError(
                        "seed analysis reuses one source path across cells"
                    )
                source_paths.add(path)
                _strict_digest(
                    source[digest_field],
                    location=f"{location}.sources.{seed}.{digest_field}",
                )

        summary = cell["summary"]
        if not isinstance(summary, dict) or set(summary) != {"raw_aurc", "contrasts"}:
            raise ValueError(f"{location}.summary has an unexpected schema")
        raw = summary["raw_aurc"]
        if not isinstance(raw, dict) or set(raw) != risk_names:
            raise ValueError(f"{location}.summary.raw_aurc is incomplete")
        for risk, methods in raw.items():
            if not isinstance(methods, dict) or set(methods) != method_names:
                raise ValueError(f"{location}.summary.raw_aurc.{risk} is incomplete")
            for method, method_summary in methods.items():
                _validate_three_seed_summary(
                    method_summary,
                    location=f"{location}.summary.raw_aurc.{risk}.{method}",
                    contrast=False,
                )
        contrasts = summary["contrasts"]
        if not isinstance(contrasts, dict) or set(contrasts) != set(contrast_by_name):
            raise ValueError(f"{location}.summary.contrasts is incomplete")
        for name, contrast_summary in contrasts.items():
            values = _validate_three_seed_summary(
                contrast_summary,
                location=f"{location}.summary.contrasts.{name}",
                contrast=True,
            )
            contrast = contrast_by_name[name]
            for seed in ("0", "1", "2"):
                expected = (
                    raw[contrast.risk][contrast.left]["values"][seed]
                    - raw[contrast.risk][contrast.right]["values"][seed]
                )
                if not math.isclose(
                    values[seed], expected, rel_tol=1e-15, abs_tol=1e-15
                ):
                    raise ValueError(
                        f"{location}.summary.contrasts.{name}.values.{seed} "
                        "differs from the raw AURCs"
                    )
        by_key[key] = cell
    expected_keys = {
        (dataset, condition)
        for dataset in TARGET_DATASETS
        for condition in TARGET_CONDITIONS
    }
    if set(by_key) != expected_keys:
        raise ValueError("seed-analysis dataset/condition grid is incomplete")
    expected_gate = _gate_c(cells)
    if result["gate_c"] != expected_gate:
        raise ValueError("Gate C decision is inconsistent with seed contrasts")
    return by_key


def _three_seed_summary(values):
    if set(values) != {0, 1, 2}:
        raise ValueError("seed summary requires exactly seeds 0, 1, and 2")
    ordered = [float(values[seed]) for seed in (0, 1, 2)]
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("seed summary contains a non-finite value")
    return {
        "values": {str(seed): ordered[seed] for seed in (0, 1, 2)},
        "mean": float(statistics.fmean(ordered)),
        "minimum": min(ordered),
        "maximum": max(ordered),
        "range": max(ordered) - min(ordered),
        "sample_standard_deviation": float(statistics.stdev(ordered)),
    }


def _contrast_seed_summary(values):
    summary = _three_seed_summary(values)
    signs = {str(seed): _direction(values[seed]) for seed in (0, 1, 2)}
    counts = {sign: tuple(signs.values()).count(sign) for sign in (-1, 0, 1)}
    majority = [sign for sign, count in counts.items() if count >= 2]
    majority_direction = majority[0] if len(majority) == 1 else None
    summary.update(
        {
            "directions": signs,
            "majority_direction": majority_direction,
            "seed0_is_majority_direction": (
                majority_direction is not None and signs["0"] == majority_direction
            ),
            "direction_reversal": -1 in signs.values() and 1 in signs.values(),
        }
    )
    return summary


def _gate_c(cells):
    seed0_not_majority = []
    reversal_counts = {contrast.name: 0 for contrast in CONTRASTS}
    for cell in cells:
        for name, summary in cell["summary"]["contrasts"].items():
            if not summary["seed0_is_majority_direction"]:
                seed0_not_majority.append(
                    {
                        "dataset": cell["dataset"],
                        "condition": cell["condition"],
                        "contrast": name,
                    }
                )
            if summary["direction_reversal"]:
                reversal_counts[name] += 1
    reversal_threshold = [name for name, count in reversal_counts.items() if count >= 3]
    fired = bool(seed0_not_majority or reversal_threshold)
    reasons = []
    if seed0_not_majority:
        reasons.append("seed0_not_majority_direction")
    if reversal_threshold:
        reasons.append("at_least_three_conditions_reverse_for_one_contrast")
    return {
        "fired": fired,
        "decision": (
            "move three-seed table to the main results and call the affected "
            "comparison training-sensitive"
            if fired
            else "report direction retention in the main text; keep full values in appendix"
        ),
        "seed0_not_majority_cells": seed0_not_majority,
        "direction_reversal_counts": reversal_counts,
        "contrasts_with_at_least_three_reversals": reversal_threshold,
        "reasons": reasons,
    }


def analyze_seed_extension(
    downstream_binding,
    *,
    canonical_analysis,
    expected_canonical_analysis_sha256,
):
    runs, canonical_provenance = validate_analysis_inputs(
        downstream_binding,
        canonical_analysis=canonical_analysis,
        expected_canonical_analysis_sha256=expected_canonical_analysis_sha256,
    )
    keys = sorted(runs[0])
    cells = []
    for dataset, condition in keys:
        summaries = {
            seed: _summarize_condition(runs[seed][(dataset, condition)])
            for seed in (0, 1, 2)
        }
        raw_summary = {
            risk: {
                method: _three_seed_summary(
                    {
                        seed: summaries[seed]["raw_aurc"][risk][method]
                        for seed in (0, 1, 2)
                    }
                )
                for method, _ in METHODS
            }
            for risk, _ in RISKS
        }
        contrast_summary = {
            contrast.name: _contrast_seed_summary(
                {
                    seed: summaries[seed]["contrasts"][contrast.name]
                    for seed in (0, 1, 2)
                }
            )
            for contrast in CONTRASTS
        }
        sources = {
            str(seed): {
                "records": runs[seed][(dataset, condition)].jsonl_path.as_posix(),
                "records_sha256": runs[seed][(dataset, condition)].manifest[
                    "jsonl_sha256"
                ],
                "manifest": runs[seed][(dataset, condition)].manifest_path.as_posix(),
                "manifest_sha256": _sha256(
                    runs[seed][(dataset, condition)].manifest_path
                ),
            }
            for seed in (0, 1, 2)
        }
        cells.append(
            {
                "dataset": dataset,
                "condition": condition,
                "model": runs[0][(dataset, condition)].manifest["model"],
                "num_images_per_seed": len(runs[0][(dataset, condition)].rows),
                "sources": sources,
                "summary": {
                    "raw_aurc": raw_summary,
                    "contrasts": contrast_summary,
                },
            }
        )
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis": {
            "estimand": (
                "descriptive target-model training-seed variation over seeds 0,1,2"
            ),
            "replication_unit": "one independently trained checkpoint",
            "inference": "none; no image pooling and no seed-level hypothesis test",
            "statistics": "three values, mean, range, and sample standard deviation",
            "aurc_scale": "raw [0,1]; renderers may display 100 x AURC",
            "contrast_definition": "AURC(left score) - AURC(right score)",
            "contrast_definitions": [asdict(contrast) for contrast in CONTRASTS],
            "cohort_join_fields": list(COHORT_JOIN_FIELDS),
        },
        "provenance": {
            "downstream_lock": {
                "path": downstream_binding["path"].as_posix(),
                "sha256": downstream_binding["sha256"],
            },
            "canonical_seed0": canonical_provenance,
            "analysis_source_sha256": _analysis_source_sha256(),
        },
        "cells": cells,
    }
    result["gate_c"] = _gate_c(cells)
    return result


def main(argv=None):
    args = parse_args(argv)
    downstream = load_downstream_lock(
        args.downstream_lock,
        expected_sha256=args.expected_downstream_lock_sha256,
    )
    result = analyze_seed_extension(
        downstream,
        canonical_analysis=args.canonical_analysis,
        expected_canonical_analysis_sha256=(args.expected_canonical_analysis_sha256),
    )
    validate_analysis_document(result)
    output = Path(args.output)
    expected_output = (
        Path(downstream["binding"]["spec"]["paths"]["analysis_root"]) / "analysis.json"
    )
    if output != expected_output:
        raise ValueError(f"seed analysis must be written to {expected_output}")
    _atomic_write_new(output, result)
    print(f"saved {output}")
    print(f"analysis_sha256={_sha256(output)}")
    print(f"gate_c_fired={str(result['gate_c']['fired']).lower()}")
    return output


if __name__ == "__main__":
    main()
