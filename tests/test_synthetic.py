"""Tests for the locked one-cell-per-job known-posterior study."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze.synthetic import analyze
from scripts.render.synthetic import hd95_contamination_curve, main as render_main
from scripts.submit import synthetic as submit_synthetic
from selectseg.studies.synthetic import (
    COUPLINGS,
    LOSSES,
    M_VALUES,
    Cell,
    all_cells,
    cell_seeds,
    dice_loss,
    exact_dice_q_risk,
    exact_total_variation,
    jaccard_distance,
    load_synthetic_lock,
    pilot_cells,
    sample_p_q,
    selected_cells,
    simulate_cell,
)
from selectseg.studies.synthetic_matrix import (
    ESTIMATOR_COUPLINGS,
    _load_config as load_matrix_config,
    all_cells as all_matrix_cells,
    simulate_cell as simulate_matrix_cell,
)
from selectseg.quadrature import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_repository_lock_and_job_counts_are_exact():
    binding = load_synthetic_lock(ROOT / submit_synthetic.DEFAULT_LOCK)
    assert len(all_cells(binding["spec"])) == 360
    assert len(pilot_cells(binding["spec"])) == 12
    assert len(selected_cells(binding["spec"], "full")) == 348
    pilot = submit_synthetic.plan_synthetic_jobs(
        ROOT / submit_synthetic.DEFAULT_LOCK, phase="pilot"
    )
    assert len(pilot) == 12
    with pytest.raises(ValueError, match="requires explicit --pilot-analysis"):
        submit_synthetic.plan_synthetic_jobs(
            ROOT / submit_synthetic.DEFAULT_LOCK, phase="full"
        )
    for job in pilot:
        command = list(job.command)
        assert command.count("scripts/slurm/run.sbatch") == 1
        assert command.count("--coupling") == 1
        assert command.count("--sharpness") == 1
        assert command.count("--morphology") == 1
        assert command.count("--replicate") == 1
        assert "--array" not in command
        assert "--gres" not in command


def test_matrix_config_and_scheduler_cover_all_estimator_couplings():
    config_path, config = load_matrix_config(
        ROOT / submit_synthetic.DEFAULT_MATRIX_CONFIG
    )
    assert config_path.is_file()
    assert len(all_matrix_cells(config)) == 360
    pilot = submit_synthetic.plan_matrix_jobs(config_path, phase="pilot")
    full = submit_synthetic.plan_matrix_jobs(config_path, phase="full")
    assert len(pilot) == 12
    assert len(full) == 348
    for job in pilot:
        command = list(job.command)
        assert "--true-coupling" in command
        assert "--array" not in command
        assert set(command[command.index("--partition") + 1].split(",")) == {
            "saffo-2tb",
            "agsmall",
            "amdsmall",
            "msismall",
        }


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_all_couplings_preserve_the_same_pixel_marginals(coupling):
    probability = np.array([[0.08, 0.25, 0.55], [0.72, 0.87, 0.96]])
    global_uniforms = np.random.default_rng(11).random(60_000)
    p_masks, q_masks = sample_p_q(
        probability,
        coupling,
        global_uniforms,
        np.random.default_rng(29),
        block_size=1,
    )
    assert np.max(np.abs(p_masks.mean(axis=0) - probability)) < 0.008
    assert np.max(np.abs(q_masks.mean(axis=0) - probability)) < 0.008


def test_exact_dice_integral_and_mask_distances():
    probability = np.array([[0.1, 0.2], [0.8, 0.9]])
    action = probability >= 0.5
    exact = exact_dice_q_risk(probability, action)
    nodes = (np.arange(200_000, dtype=float) + 0.5) / 200_000
    dense = np.mean([dice_loss(probability >= node, action) for node in nodes])
    assert exact == pytest.approx(dense, abs=1e-6)
    assert dice_loss(action, action) == 0.0
    assert jaccard_distance(action, action) == 0.0
    assert dice_loss(np.zeros_like(action), np.zeros_like(action)) == 0.0


def test_matrix_matched_coupling_recovers_the_monte_carlo_truth():
    _, config = load_matrix_config(ROOT / submit_synthetic.DEFAULT_MATRIX_CONFIG)
    config = copy.deepcopy(config)
    config["protocol"].update(
        {
            "height": 8,
            "width": 8,
            "cohort_size": 3,
            "posterior_draws": 16,
            "mc_batches": 4,
            "workers_per_job": 1,
            "local_block_size": 4,
            "sharpness_pixels": {"diffuse": 3.0, "medium": 1.5, "sharp": 0.7},
        }
    )
    config["protocol"]["spatial_copula"]["posterior_batch_size"] = 4
    cell = Cell("independent_bernoulli", "medium", "disk", 0)
    result = simulate_matrix_cell(config, cell, workers=1)
    assert set(result["losses"]) == set(LOSSES)
    for loss in LOSSES:
        estimators = result["losses"][loss]["estimators"]
        assert set(estimators) == set(ESTIMATOR_COUPLINGS)
        matched = estimators["independent_bernoulli"]
        assert matched["risk_error"]["maximum"] == 0.0
        assert matched["aurc_regret"] == pytest.approx(0.0)


def test_exact_tv_detects_coupling_misspecification():
    probability = np.array([[0.1, 0.2], [0.8, 0.9]])
    assert exact_total_variation(probability, "shared_threshold", 1) == 0.0
    for coupling in COUPLINGS[1:]:
        assert 0.0 < exact_total_variation(probability, coupling, 1) <= 1.0


def test_small_cell_retains_only_aggregate_statistics():
    binding = load_synthetic_lock(ROOT / submit_synthetic.DEFAULT_LOCK)
    spec = copy.deepcopy(binding["spec"])
    spec["protocol"].update(
        {
            "height": 16,
            "width": 16,
            "cohort_size": 3,
            "posterior_draws": 16,
            "mc_batches": 4,
        }
    )
    cell = Cell("shared_threshold", "medium", "disk", 0)
    summary = simulate_cell(spec, cell)
    threaded = simulate_cell(spec, cell, workers=2)
    assert summary["cohort"]["size"] == 3
    assert summary["cohort"]["posterior_draws_per_image"] == 16
    assert set(summary["losses"]) == set(LOSSES)
    assert set(summary["losses"]["dice"]["estimators"]) == {
        *(f"m{value}" for value in M_VALUES),
        "exact",
    }
    serialized = json.dumps(summary)
    assert "p_masks" not in serialized
    assert "q_masks" not in serialized
    assert summary["posterior_discrepancy"]["total_variation_exact"]["maximum"] == 0.0
    assert summary["losses"]["nhd95"]["loss_pushforward_w1_empirical"]["mean"] == 0.0
    assert "no mask-Wasserstein corollary" in summary["monte_carlo_note"]
    assert threaded == summary


def _summary(cell, coupling_index, sharpness_index):
    estimators = {}
    for loss_index, loss in enumerate(LOSSES):
        rows = {}
        names = ["m2", "m8", "m32", "m128"] + (["exact"] if loss == "dice" else [])
        for estimator_index, name in enumerate(names):
            value = (
                0.0
                if cell.coupling == "shared_threshold"
                else (
                    0.01
                    * (
                        coupling_index
                        + sharpness_index
                        + loss_index
                        + estimator_index
                        + 1
                    )
                )
            )
            rows[name] = {
                "score_error": {
                    "mean": value,
                    "standard_deviation": 0.0,
                    "median": value,
                    "q95": value,
                    "maximum": value,
                    "signed_bias": value,
                },
                "spearman_risk_ranking": 1.0 - value,
                "kendall_tau_b_risk_ranking": 1.0 - value,
                "aurc": 0.2 + value,
                "oracle_aurc": 0.2,
                "aurc_regret": value,
                "aurc_regret_mc_se": 0.001,
            }
        estimators[loss] = {
            "true_p_risk": {"mean": 0.2},
            "q_monte_carlo_risk": {"mean": 0.2},
            "cell_mean_true_risk_mc_se": 0.01,
            "cell_mean_qmc_minus_true": 0.0,
            "estimators": rows,
            "loss_pushforward_w1_empirical": {
                "mean": value,
                "mean_mc_se": 0.001,
                "maximum": value,
            },
        }
    zero = 0.0 if cell.coupling == "shared_threshold" else 0.2
    return {
        "summary_schema_version": 1,
        "cell": {
            "coupling": cell.coupling,
            "sharpness": cell.sharpness,
            "morphology": cell.morphology,
            "replicate": cell.replicate,
        },
        "seeds": {},
        "cohort": {"size": 24},
        "posterior_discrepancy": {
            "total_variation_exact": {"mean": zero, "maximum": zero},
            "empty_event_tv_lower_bound_exact": {"mean": zero},
            "paired_jaccard_transport_cost_upper_bound": {
                "mean": zero,
                "maximum": zero,
            },
            "paired_normalized_hd_transport_cost_upper_bound": {
                "mean": zero,
                "maximum": zero,
            },
        },
        "losses": estimators,
        "monte_carlo_note": "test",
    }


def _write_fake_pilot(tmp_path):
    repository = load_synthetic_lock(ROOT / submit_synthetic.DEFAULT_LOCK)
    spec = copy.deepcopy(repository["spec"])
    spec["paths"] = {
        "pilot_output_root": str(tmp_path / "pilot"),
        "main_output_root": str(tmp_path / "main"),
        "analysis_output_root": str(tmp_path / "analysis"),
    }
    spec_path = tmp_path / "synthetic.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    code_sources = [
        {"path": str(path), "sha256": sha} for path, sha in repository["code_sources"]
    ]
    lock = {
        "lock_schema_version": 1,
        "campaign_id": spec["campaign_id"],
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "code_sources": code_sources,
    }
    lock_path = tmp_path / "synthetic.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    binding = load_synthetic_lock(lock_path)
    for coupling_index, coupling in enumerate(spec["grid"]["couplings"]):
        for sharpness_index, sharpness in enumerate(spec["grid"]["sharpness_levels"]):
            cell = Cell(coupling, sharpness, "disk", 0)
            summary = _summary(cell, coupling_index, sharpness_index)
            summary["seeds"] = cell_seeds(spec, cell)
            artifact_id = f"{coupling_index:02d}{sharpness_index:02d}" + "a" * 60
            directory = (
                Path(spec["paths"]["pilot_output_root"])
                / coupling
                / sharpness
                / "disk"
                / "replicate-00"
                / artifact_id
            )
            directory.mkdir(parents=True)
            summary_path = directory / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            manifest = {
                "manifest_schema_version": 1,
                "artifact_type": "selectseg.synthetic_cell",
                "artifact_id": artifact_id,
                "campaign_id": spec["campaign_id"],
                "created_utc": "2026-07-19T00:00:00+00:00",
                "phase": "pilot",
                "cell": summary["cell"],
                "seeds": summary["seeds"],
                "lock": {"path": str(lock_path), "sha256": binding["sha256"]},
                "spec": {"path": str(spec_path), "sha256": binding["spec_sha256"]},
                "code_sources": code_sources,
                "summary": {
                    "path": "summary.json",
                    "sha256": sha256_file(summary_path),
                },
                "runtime_seconds": 15.0,
                "environment": {},
                "command": ["pytest"],
                "storage_policy": "aggregate sufficient statistics only; no masks persisted",
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    return lock_path


def _write_pilot_analysis(lock, output):
    result = analyze(lock, mode="pilot")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return output, sha256_file(output), result


def test_full_plan_requires_and_revalidates_passing_pilot_gate(tmp_path):
    lock = _write_fake_pilot(tmp_path)
    analysis_path, digest, _ = _write_pilot_analysis(
        lock, tmp_path / "pilot-analysis.json"
    )
    full = submit_synthetic.plan_synthetic_jobs(
        lock,
        phase="full",
        pilot_analysis=analysis_path,
        expected_pilot_analysis_sha256=digest,
    )
    assert len(full) == 348
    assert len({job.key for job in full}) == 348
    for job in full:
        command = list(job.command)
        assert command.count("scripts/slurm/run.sbatch") == 1
        assert command.count("--phase") == 1
        assert command[command.index("--phase") + 1] == "full"
        assert "--array" not in command


def test_full_gate_rejects_wrong_hash_incomplete_ids_and_stale_sources(tmp_path):
    lock = _write_fake_pilot(tmp_path)
    analysis_path, digest, analysis = _write_pilot_analysis(
        lock, tmp_path / "pilot-analysis.json"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        submit_synthetic.plan_synthetic_jobs(
            lock,
            phase="full",
            pilot_analysis=analysis_path,
            expected_pilot_analysis_sha256="0" * 64,
        )

    incomplete = copy.deepcopy(analysis)
    incomplete["source_manifests"].pop()
    incomplete_path = tmp_path / "incomplete-analysis.json"
    incomplete_path.write_text(json.dumps(incomplete, indent=2) + "\n")
    with pytest.raises(ValueError, match="12 distinct source manifests"):
        submit_synthetic.plan_synthetic_jobs(
            lock,
            phase="full",
            pilot_analysis=incomplete_path,
            expected_pilot_analysis_sha256=sha256_file(incomplete_path),
        )

    manifest_path = Path(analysis["source_manifests"][0]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_seconds"] += 1.0
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="differs from strict recomputation"):
        submit_synthetic.plan_synthetic_jobs(
            lock,
            phase="full",
            pilot_analysis=analysis_path,
            expected_pilot_analysis_sha256=digest,
        )


def test_full_gate_rejects_a_strict_but_failed_runtime_gate(tmp_path):
    lock = _write_fake_pilot(tmp_path)
    binding = load_synthetic_lock(lock)
    cell = pilot_cells(binding["spec"])[0]
    cell_root = (
        Path(binding["spec"]["paths"]["pilot_output_root"])
        / cell.coupling
        / cell.sharpness
        / cell.morphology
        / "replicate-00"
    )
    manifest_path = next(cell_root.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_seconds"] = 10_801.0
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    analysis_path, digest, result = _write_pilot_analysis(
        lock, tmp_path / "failed-pilot-analysis.json"
    )
    assert result["pilot_gate"]["passed"] is False
    with pytest.raises(ValueError, match="pilot gate did not pass"):
        submit_synthetic.plan_synthetic_jobs(
            lock,
            phase="full",
            pilot_analysis=analysis_path,
            expected_pilot_analysis_sha256=digest,
        )


def test_full_submission_uses_only_fixed_append_only_receipt(tmp_path, monkeypatch):
    lock = _write_fake_pilot(tmp_path)
    analysis_path, digest, _ = _write_pilot_analysis(
        lock, tmp_path / "pilot-analysis.json"
    )
    common = [
        "--lock",
        str(lock),
        "--phase",
        "full",
        "--pilot-analysis",
        str(analysis_path),
        "--expected-pilot-analysis-sha256",
        digest,
    ]
    with pytest.raises(ValueError, match="fixed receipt"):
        submit_synthetic.main(
            [*common, "--submit", "--receipt", str(tmp_path / "new.jsonl")]
        )
    with pytest.raises(ValueError, match="only together"):
        submit_synthetic.main([*common, "--receipt", str(tmp_path / "new.jsonl")])

    observed = {}

    def fake_execute(jobs, *, submit, receipt_path):
        observed.update(jobs=jobs, submit=submit, receipt_path=receipt_path)
        return ()

    monkeypatch.setattr(submit_synthetic, "execute_plan", fake_execute)
    submit_synthetic.main(
        [
            *common,
            "--submit",
            "--receipt",
            submit_synthetic.FULL_RECEIPT.as_posix(),
        ]
    )
    assert len(observed["jobs"]) == 348
    assert observed["submit"] is True
    assert (
        observed["receipt_path"]
        == (submit_synthetic.REPO_ROOT / submit_synthetic.FULL_RECEIPT).resolve()
    )


def test_strict_pilot_analysis_and_renderer(tmp_path):
    lock = _write_fake_pilot(tmp_path)
    result = analyze(lock, mode="pilot")
    assert result["num_cells"] == 12
    assert result["pilot_gate"]["passed"] is True
    assert len(result["headline_groups"]) == 12
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2) + "\n")
    output = tmp_path / "rendered"
    render_main(["--analysis", str(analysis_path), "--output-dir", str(output)])
    tex = (output / "synthetic_posterior_summary.tex").read_text()
    assert "Known-posterior stress test" in tex
    assert "no HD95 mask-Wasserstein corollary" in tex
    assert "AURC regret is multiplied by 100 for display only" in tex
    assert r"$100\times$AURC reg." in tex
    assert r"\begin{table*}[t]" in tex
    assert r"\resizebox{\textwidth}{!}{%" in tex
    assert "posterior-draw Monte Carlo SE" in tex
    assert "integration SE" not in tex
    # The first non-shared M32 Dice regret is 0.05 in the fixture and is
    # therefore displayed as 5.0000 without changing the analysis JSON.
    assert "Independent Bernoulli & 0.0500 & 5.0000" in tex
    assert (output / "synthetic_posterior_summary.pdf").is_file()
    assert (output / "synthetic_posterior_render.manifest.json").is_file()


def test_hd95_contamination_curve_exposes_the_five_percent_transition():
    rows = hd95_contamination_curve()
    below = max(
        row["hd95_pixels"]
        for row in rows
        if row["observed_contamination_percent"] < 5
    )
    above = min(
        row["hd95_pixels"]
        for row in rows
        if row["observed_contamination_percent"] > 5.1
    )
    assert below == 0
    assert above > 90
    assert all(row["hd_pixels"] == 100 for row in rows[1:])
    assert rows[-1]["dice_loss"] < 0.01


def test_analyzer_rejects_duplicate_published_cell(tmp_path):
    lock = _write_fake_pilot(tmp_path)
    binding = load_synthetic_lock(lock)
    cell = pilot_cells(binding["spec"])[0]
    root = (
        Path(binding["spec"]["paths"]["pilot_output_root"])
        / cell.coupling
        / cell.sharpness
        / cell.morphology
        / "replicate-00"
    )
    (root / ("f" * 64)).mkdir()
    with pytest.raises(ValueError, match="exactly one published artifact"):
        analyze(lock, mode="pilot")
