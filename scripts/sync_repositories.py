"""Safely mirror the public paper and focused code through explicit allowlists.

The script never deletes unmanaged destination files, stages changes, commits,
pushes, or reads Git remotes.  To publish a changed generated figure bundle
fail closed, ``--apply`` temporarily withdraws that bundle's exact allowlisted
TeX guard, writes payloads and manifest, then restores the guard last.  A dry
run is the default.

Examples
--------
Preview both mirrors::

    python -m scripts.sync_repositories

Update only the local Overleaf clone::

    python -m scripts.sync_repositories --target overleaf --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from selectseg.scientific_inputs import load_root_lock


REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly the six three-loss tables emitted by ``scripts.render_paper_tables``.
FINAL_GENERATED_TABLE_FILES = (
    "Tables/main_results.tex",
    "Tables/full_target_results.tex",
    "Tables/complete_results.tex",
    "Tables/cross_loss_results.tex",
    "Tables/quadrature_ablation.tex",
    "Tables/statistical_tests.tex",
)
# The renderer publishes this last, after all six tables have been written and
# verified to carry the same source hash.  Mirroring the sentinel is necessary:
# the manuscript intentionally refuses to input any final result without it.
RESULTS_COMPLETION_SENTINEL_FILE = "Tables/results_complete.tex"

# Binary PDF artwork is source material, unlike LaTeX build products.  Keep the
# list explicit so the public mirrors cannot acquire arbitrary local renders.
MANUSCRIPT_PDF_FILES = ("Figures/loss_indexed_framework.pdf",)
# Dataset-level risk--coverage curves are generated only after the locked
# campaign is assembled.  They are individually enumerated and guarded so a
# pre-results mirror remains valid without silently admitting other PDFs.
OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES = (
    "Figures/risk_coverage_all_indexed_pet.pdf",
    "Figures/risk_coverage_all_indexed_kvasir.pdf",
    "Figures/risk_coverage_all_indexed_fives.pdf",
    "Figures/risk_coverage_all_indexed_isic.pdf",
    "Figures/risk_coverage_all_indexed_tn3k.pdf",
)
OPTIONAL_GENERATED_FIGURE_SENTINEL_FILES = ("Figures/risk_coverage_complete.tex",)
OPTIONAL_GENERATED_FIGURE_MANIFEST_FILES = ("Figures/risk_coverage_manifest.json",)
# Explicitly allowlisted but guarded generated inputs. They join a mirror only
# after their strict producer has created them; absence is the manuscript's
# intentional pre-results state, not an invitation to discover arbitrary files.
OPTIONAL_GENERATED_MANUSCRIPT_FILES = (
    "Tables/binary_diagnostics.tex",
    "Tables/cardinality_diagnostics.tex",
    "Tables/gamma_sensitivity.tex",
    "Tables/working_risk_diagnostics.tex",
    "Tables/m128_numerical_reference.tex",
    "Tables/binary_runtime.tex",
    "Tables/seed_robustness.tex",
    "Tables/seed_sensitivity_main.tex",
    "Tables/synthetic_posterior_summary.tex",
    "Figures/synthetic_posterior_summary.pdf",
    "Figures/matched_risk_reliability_pet.pdf",
    "Figures/matched_risk_reliability_kvasir.pdf",
    "Figures/matched_risk_reliability_fives.pdf",
    "Figures/matched_risk_reliability_isic.pdf",
    "Figures/matched_risk_reliability_tn3k.pdf",
)
OPTIONAL_GENERATED_QUALITATIVE_FILES = (
    "Figures/qualitative_cases.tex",
    "Figures/qualitative_manifest.json",
    "Figures/qualitative_pet.png",
    "Figures/qualitative_kvasir.png",
    "Figures/qualitative_fives.png",
    "Figures/qualitative_isic.png",
    "Figures/qualitative_tn3k.png",
)

# Optional seed-extension release.  Private execution records remain excluded;
# the only row-level additions are the explicitly path-free 30-condition replay
# bundle, validation lock, and write-last completion guard.
PUBLIC_SEED_REPLAY_RUNS = (
    (0, "fives", "clipseg-target", "627922c22aa3aa21"),
    (0, "fives", "deeplabv3-target", "a62d754378787146"),
    (0, "isic", "clipseg-target", "69fc681ff737f148"),
    (0, "isic", "deeplabv3-target", "a42288ea78a340b1"),
    (0, "kvasir", "clipseg-target", "1a1c02b968ab86a4"),
    (0, "kvasir", "deeplabv3-target", "5cca940975b7f02d"),
    (0, "pet", "clipseg-target", "cd4341aaed2bda20"),
    (0, "pet", "deeplabv3-target", "fd2f61c609c18fba"),
    (0, "tn3k", "clipseg-target", "eccb8f4f045a5473"),
    (0, "tn3k", "deeplabv3-target", "01b4c3a58986cf27"),
    (1, "fives", "clipseg-target", "2435c524ef0766f3"),
    (1, "fives", "deeplabv3-target", "9f314f06a2eab595"),
    (1, "isic", "clipseg-target", "c848014159175786"),
    (1, "isic", "deeplabv3-target", "46dc7182fd5fc6e3"),
    (1, "kvasir", "clipseg-target", "2860e2e7601882f3"),
    (1, "kvasir", "deeplabv3-target", "acc2da1ce32ad656"),
    (1, "pet", "clipseg-target", "0fec2bd382f23d4f"),
    (1, "pet", "deeplabv3-target", "127e78caea1a1586"),
    (1, "tn3k", "clipseg-target", "63176cdfefc0aa5f"),
    (1, "tn3k", "deeplabv3-target", "f7530ce073e35ec3"),
    (2, "fives", "clipseg-target", "3c23e51d0dbd3f7c"),
    (2, "fives", "deeplabv3-target", "164b183d9f7e8b7b"),
    (2, "isic", "clipseg-target", "1bef05e564bac5ac"),
    (2, "isic", "deeplabv3-target", "db6ed7fad1d713ac"),
    (2, "kvasir", "clipseg-target", "6b50e05a2e7b7bb2"),
    (2, "kvasir", "deeplabv3-target", "7c7a5f46a399fb83"),
    (2, "pet", "clipseg-target", "4f23a8a500f3ad40"),
    (2, "pet", "deeplabv3-target", "42833caf0bdfc194"),
    (2, "tn3k", "clipseg-target", "73f6639aefd90529"),
    (2, "tn3k", "deeplabv3-target", "2b5e117d5edba5b6"),
)
OPTIONAL_PUBLIC_SEED_REPLAY_PAYLOAD_FILES: tuple[tuple[str, str], ...] = tuple(
        (
            (
                f"outputs/public_seed/seed_records/seed-{seed}/{dataset}/"
                f"{condition}/{run_id}/{name}"
            ),
            (
                f"results/seed_records/seed-{seed}/{dataset}/{condition}/"
                f"{run_id}/{name}"
            ),
        )
        for seed, dataset, condition, run_id in PUBLIC_SEED_REPLAY_RUNS
        for name in ("manifest.json", "records.jsonl")
    )
OPTIONAL_PUBLIC_SEED_REPLAY_FILES: tuple[tuple[str, str], ...] = (
    *OPTIONAL_PUBLIC_SEED_REPLAY_PAYLOAD_FILES,
    (
        "outputs/public_seed/seed_replay.lock.json",
        "results/seed_replay.lock.json",
    ),
    (
        "outputs/public_seed/seed_replay.complete.json",
        "results/seed_replay.complete.json",
    ),
)
OPTIONAL_PUBLIC_SEED_RESULT_FILES: tuple[tuple[str, str], ...] = (
    (
        "outputs/public_seed/seed_robustness_analysis.json",
        "results/seed_robustness_analysis.json",
    ),
    (
        "outputs/public_seed/seed_scheduler_summary.json",
        "results/seed_scheduler_summary.json",
    ),
    (
        "outputs/public_seed/seed_provenance.json",
        "results/seed_provenance.json",
    ),
)
_PUBLIC_SEED_ANALYSIS_SOURCE = OPTIONAL_PUBLIC_SEED_RESULT_FILES[0][0]
_PUBLIC_SEED_SCHEDULER_SOURCE = OPTIONAL_PUBLIC_SEED_RESULT_FILES[1][0]
_PUBLIC_SEED_GUARD_SOURCE = OPTIONAL_PUBLIC_SEED_RESULT_FILES[2][0]
_PUBLIC_SEED_REPLAY_LOCK_SOURCE = OPTIONAL_PUBLIC_SEED_REPLAY_FILES[-2][0]
_PUBLIC_SEED_REPLAY_GUARD_SOURCE = OPTIONAL_PUBLIC_SEED_REPLAY_FILES[-1][0]

# Complete manuscript source package: reachable TeX inputs plus build notes.
# Drafts, PDFs, plans, audit notes, and docs/legacy are deliberately absent.
MANUSCRIPT_FILES = (
    "README.md",
    "main.tex",
    "references.bib",
    "iclr2026_conference.sty",
    "iclr2026_conference.bst",
    "fancyhdr.sty",
    "natbib.sty",
    "Sections/00_abstract.tex",
    "Sections/01_introduction.tex",
    "Sections/02_loss_indexed.tex",
    "Sections/03_level_set.tex",
    "Sections/04_dice_sdc.tex",
    "Sections/05_experiments.tex",
    "Sections/06_results.tex",
    "Sections/08_limitations_conclusion.tex",
    "Sections/09_statements.tex",
    "Sections/A_theory.tex",
    "Sections/B_additional_results.tex",
    "Sections/C_expanded_limitations.tex",
    "Sections/D_benchmark_protocol.tex",
    "Sections/E_llm_usage.tex",
    *FINAL_GENERATED_TABLE_FILES,
    RESULTS_COMPLETION_SENTINEL_FILE,
    *MANUSCRIPT_PDF_FILES,
)

# The public code mirror contains the focused binary workflow and only the
# conventional model/data utilities that it imports.  The old selective,
# multiclass, band, asymmetric, refutation, and scratch pipelines are absent.
GITHUB_ROOT_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "conftest.py",
)
GITHUB_PACKAGE_FILES = (
    "selectseg/__init__.py",
    "selectseg/binary_artifacts.py",
    "selectseg/binary_boundary.py",
    "selectseg/benchmark_binary_runtime.py",
    "selectseg/binary_diagnostics.py",
    "selectseg/cardinality_diagnostics.py",
    "selectseg/binary_framework.py",
    "selectseg/binary_seed_extension.py",
    "selectseg/binary_seed_downstream.py",
    "selectseg/binary_eval.py",
    "selectseg/binary_baselines.py",
    "selectseg/data.py",
    "selectseg/freeze_binary_maps.py",
    "selectseg/models.py",
    "selectseg/scientific_inputs.py",
    "selectseg/score_binary_common.py",
    "selectseg/score_binary_gamma_sensitivity.py",
    "selectseg/score_binary_m128_auxiliary.py",
    "selectseg/score_binary_simulation.py",
    "selectseg/synthetic_posterior.py",
    "selectseg/threshold_estimators.py",
    "selectseg/train.py",
)
GITHUB_SCRIPT_FILES = (
    "configs/binary_midpoint_main.json",
    "configs/binary_midpoint_main_v2.json",
    "configs/binary_midpoint_smoke_v1.json",
    "configs/binary_midpoint_pilot.json",
    "configs/binary_midpoint_dual_pilot.json",
    "configs/auxiliary/binary_gamma_sensitivity-v1.json",
    "configs/auxiliary/binary_gamma_sensitivity-v1.lock.json",
    "configs/auxiliary/binary_cardinality_diagnostics-v1.json",
    "configs/auxiliary/binary_cardinality_diagnostics-v1.lock.json",
    "configs/auxiliary/binary_runtime-v1.json",
    "configs/auxiliary/binary_runtime_ladder-v2.json",
    "configs/auxiliary/binary_runtime_ladder-v2.lock.json",
    "configs/auxiliary/binary_seed_extension-v1.json",
    "configs/auxiliary/binary_seed_extension-v1.lock.json",
    "configs/auxiliary/synthetic_posterior-v1.json",
    "configs/auxiliary/synthetic_posterior-v1.lock.json",
    "configs/auxiliary/synthetic_posterior-v1.README.md",
    "configs/estimators/midpoint-v1.json",
    "scripts/analyze_binary.py",
    "scripts/build_anonymous_analysis_artifact.py",
    "scripts/collect_binary_seed_diagnostics.py",
    "scripts/analyze_binary_seed_extension.py",
    "scripts/analyze_binary_diagnostics.py",
    "scripts/analyze_cardinality_diagnostics.py",
    "scripts/analyze_gamma_sensitivity.py",
    "scripts/analyze_m128_auxiliary.py",
    "scripts/analyze_matched_risk_reliability.py",
    "scripts/analyze_binary_runtime.py",
    "scripts/analyze_synthetic_posterior.py",
    "scripts/analyze_working_risk_diagnostics.py",
    "scripts/assemble_binary_simulations.py",
    "scripts/diagnose_binary_artifact.py",
    "scripts/download_binary_assets.py",
    "scripts/export_portable_analysis.py",
    "scripts/export_binary_seed_provenance.py",
    "scripts/export_seed_replay_bundle.py",
    "scripts/export_public_provenance.py",
    "scripts/adjust_seed_downstream_timelimits.py",
    "scripts/finalize_seed_scheduler_ledger.py",
    "scripts/merge_binary_auxiliary.py",
    "scripts/make_method_figure.py",
    "scripts/plot_risk_coverage.py",
    "scripts/render_paper_tables.py",
    "scripts/render_gamma_sensitivity.py",
    "scripts/render_m128_auxiliary.py",
    "scripts/render_matched_risk_reliability.py",
    "scripts/render_binary_runtime.py",
    "scripts/render_binary_seed_extension.py",
    "scripts/render_seed_gate_table.py",
    "scripts/replay_seed_robustness.py",
    "scripts/publish_binary_seed_extension.py",
    "scripts/render_cardinality_diagnostics.py",
    "scripts/render_synthetic_posterior.py",
    "scripts/render_binary_qualitative_cases.py",
    "scripts/render_working_risk_diagnostics.py",
    "scripts/submit_binary_simulations.py",
    "scripts/submit_scientific_input_components.py",
    "scripts/submit_cardinality_diagnostics.py",
    "scripts/submit_gamma_sensitivity.py",
    "scripts/submit_m128_auxiliary.py",
    "scripts/submit_binary_runtime.py",
    "scripts/submit_binary_runtime_ladder.py",
    "scripts/submit_binary_seed_extension.py",
    "scripts/submit_synthetic_posterior.py",
    "scripts/select_binary_qualitative_cases.py",
    "scripts/slurm/env.sh",
    "scripts/slurm/build_scientific_dataset.sbatch",
    "scripts/slurm/train.sbatch",
    "scripts/slurm/binary_eval.sbatch",
    "scripts/slurm/assemble_binary_simulations.sbatch",
    "scripts/slurm/freeze_binary_maps.sbatch",
    "scripts/slurm/diagnose_binary_artifact.sbatch",
    "scripts/slurm/score_binary_common.sbatch",
    "scripts/slurm/score_binary_gamma_sensitivity.sbatch",
    "scripts/slurm/score_binary_m128_auxiliary.sbatch",
    "scripts/slurm/benchmark_binary_runtime.sbatch",
    "scripts/slurm/cardinality_diagnostics.sbatch",
    "scripts/slurm/train_binary_seed_extension.sbatch",
    "scripts/slurm/freeze_binary_seed_extension.sbatch",
    "scripts/slurm/analyze_binary_seed_extension.sbatch",
    "scripts/slurm/render_binary_seed_extension.sbatch",
    "scripts/slurm/run_synthetic_posterior.sbatch",
    "scripts/slurm/score_binary_simulation.sbatch",
    "scripts/slurm/submit_extended_binary.sh",
    "scripts/slurm/submit_strong_baselines.sh",
)

# Portable, content-addressed pre-freeze inputs.  These JSON files contain only
# relative paths, byte counts, and hashes: raw datasets, model weights, and
# checkpoints remain external.  Keep the set explicit so adding a scientific
# component or a smoke lock requires an intentional public-release review.
GITHUB_SCIENTIFIC_INPUT_FILES = (
    "configs/scientific_inputs/binary-midpoint-main-v2/base_models.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/checkpoints.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/datasets/pet.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/datasets/kvasir.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/datasets/fives.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/datasets/isic.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/datasets/tn3k.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/environment.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/source.json",
    "configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json",
    "configs/scientific_inputs/binary-midpoint-smoke-v1/checkpoints.json",
    "configs/scientific_inputs/binary-midpoint-smoke-v1/root.lock.json",
)
GITHUB_SCIENTIFIC_ROOT_CONFIGS = (
    (
        "configs/binary_midpoint_main_v2.json",
        "configs/scientific_inputs/binary-midpoint-main-v2/root.lock.json",
    ),
    (
        "configs/binary_midpoint_smoke_v1.json",
        "configs/scientific_inputs/binary-midpoint-smoke-v1/root.lock.json",
    ),
)
GITHUB_TEST_FILES = (
    "tests/test_assemble_binary_simulations.py",
    "tests/test_binary_artifacts.py",
    "tests/test_binary_boundary.py",
    "tests/test_binary_diagnostics.py",
    "tests/test_cardinality_diagnostics.py",
    "tests/test_binary_framework.py",
    "tests/test_build_anonymous_analysis_artifact.py",
    "tests/test_collect_binary_seed_diagnostics.py",
    "tests/test_binary_theory.py",
    "tests/test_binary_eval.py",
    "tests/test_binary_baselines.py",
    "tests/test_freeze_binary_maps.py",
    "tests/test_analyze_binary.py",
    "tests/test_analyze_binary_diagnostics.py",
    "tests/test_analyze_gamma_sensitivity.py",
    "tests/test_analyze_m128_auxiliary.py",
    "tests/test_binary_runtime.py",
    "tests/test_binary_seed_extension.py",
    "tests/test_synthetic_posterior.py",
    "tests/test_binary_qualitative_cases.py",
    "tests/test_analyze_working_risk_diagnostics.py",
    "tests/test_merge_binary_auxiliary.py",
    "tests/test_data.py",
    "tests/test_download_binary_assets.py",
    "tests/test_export_portable_analysis.py",
    "tests/test_export_binary_seed_provenance.py",
    "tests/test_export_public_provenance.py",
    "tests/test_adjust_seed_downstream_timelimits.py",
    "tests/test_finalize_seed_scheduler_ledger.py",
    "tests/test_models.py",
    "tests/test_plot_risk_coverage.py",
    "tests/test_render_working_risk_diagnostics.py",
    "tests/test_render_paper_tables.py",
    "tests/test_gamma_sensitivity.py",
    "tests/test_matched_risk_reliability.py",
    "tests/test_publish_binary_seed_extension.py",
    "tests/test_render_seed_gate_table.py",
    "tests/test_replay_seed_robustness.py",
    "tests/test_score_binary_simulation.py",
    "tests/test_score_binary_m128_auxiliary.py",
    "tests/test_submit_binary_simulations.py",
    "tests/test_scientific_inputs.py",
    "tests/test_submit_scientific_input_components.py",
)
# Canonical locked-campaign results.  Every source and portable destination is
# explicit: neither the synchronizer nor this declaration discovers outputs by
# glob.  Raw submission receipts are deliberately absent because they contain
# scheduler commands, job IDs, partitions, accounts, and machine paths; their
# hashes and resolved counts are published only through public_provenance.json.
GITHUB_RESULT_FILES: tuple[tuple[str, str], ...] = (
    (
        "outputs/binary_campaign/campaign.lock.json",
        "results/campaign.lock.json",
    ),
    # Auxiliary experiment locks intentionally name this canonical path.  Keep
    # a second public destination so their checked-in commands run unchanged
    # from a clean clone; both copies come from the same immutable source.
    (
        "outputs/binary_campaign/campaign.lock.json",
        "outputs/binary_campaign/campaign.lock.json",
    ),
    (
        "outputs/binary_final_v3_analysis/analysis.json",
        "results/analysis.json",
    ),
    (
        "outputs/binary_final_v3_analysis/main_table.csv",
        "results/main_table.csv",
    ),
    (
        "outputs/binary_final_v3_analysis/public_provenance.json",
        "results/public_provenance.json",
    ),
    (
        "outputs/binary_final_v2_analysis/analysis.json",
        "results/analysis_v2.json",
    ),
    (
        "outputs/binary_final_v2_analysis/public_provenance.json",
        "results/public_provenance_v2.json",
    ),
    (
        "outputs/binary_final_v2_diagnostics/diagnostics_analysis.json",
        "results/diagnostics_analysis.json",
    ),
    (
        "outputs/binary_working_risk_diagnostics_v2/diagnostics.json",
        "results/working_risk_diagnostics.json",
    ),
    (
        "outputs/binary_m128_auxiliary_analysis/analysis.json",
        "results/m128_numerical_reference.json",
    ),
    (
        "outputs/binary_qualitative_cases/f618c91bfeaa467e/selection.json",
        "results/qualitative_selection.json",
    ),
    (
        "outputs/binary_gamma_sensitivity_analysis/b0d4468443fcc46e/analysis.json",
        "results/gamma_sensitivity.json",
    ),
    (
        "outputs/binary_cardinality_diagnostics_analysis/analysis.json",
        "results/cardinality_diagnostics.json",
    ),
    (
        "outputs/binary_matched_risk_reliability/analysis.json",
        "results/matched_risk_reliability.json",
    ),
    (
        "outputs/public_auxiliary/runtime_v1_analysis.json",
        "results/runtime_v1_analysis.json",
    ),
    (
        "outputs/public_auxiliary/runtime_v1_export.json",
        "results/runtime_v1_export.json",
    ),
    (
        "outputs/public_auxiliary/runtime_ladder_v2_analysis.json",
        "results/runtime_ladder_v2_analysis.json",
    ),
    (
        "outputs/public_auxiliary/runtime_ladder_v2_export.json",
        "results/runtime_ladder_v2_export.json",
    ),
    (
        "outputs/public_auxiliary/synthetic_posterior_analysis.json",
        "results/synthetic_posterior_analysis.json",
    ),
    (
        "outputs/public_auxiliary/synthetic_posterior_export.json",
        "results/synthetic_posterior_export.json",
    ),
    (
        "outputs/public_auxiliary/synthetic_posterior_pilot_analysis.json",
        "results/synthetic_posterior_pilot_analysis.json",
    ),
    (
        "outputs/public_auxiliary/synthetic_posterior_pilot_export.json",
        "results/synthetic_posterior_pilot_export.json",
    ),
    (
        "outputs/binary_train/fives/clipseg/seed-0/train_config.json",
        "results/training/fives/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/fives/deeplabv3/seed-0/train_config.json",
        "results/training/fives/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/isic/clipseg/seed-0/train_config.json",
        "results/training/isic/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/isic/clipseg/seed-0/history.json",
        "results/training/isic/clipseg/seed-0/history.json",
    ),
    (
        "outputs/binary_train/isic/deeplabv3/seed-0/train_config.json",
        "results/training/isic/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/isic/deeplabv3/seed-0/history.json",
        "results/training/isic/deeplabv3/seed-0/history.json",
    ),
    (
        "outputs/binary_train/kvasir/clipseg/seed-0/train_config.json",
        "results/training/kvasir/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/kvasir/deeplabv3/seed-0/train_config.json",
        "results/training/kvasir/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/pet/clipseg/seed-0/train_config.json",
        "results/training/pet/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/pet/deeplabv3/seed-0/train_config.json",
        "results/training/pet/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/tn3k/clipseg/seed-0/train_config.json",
        "results/training/tn3k/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/tn3k/clipseg/seed-0/history.json",
        "results/training/tn3k/clipseg/seed-0/history.json",
    ),
    (
        "outputs/binary_train/tn3k/deeplabv3/seed-0/train_config.json",
        "results/training/tn3k/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/tn3k/deeplabv3/seed-0/history.json",
        "results/training/tn3k/deeplabv3/seed-0/history.json",
    ),
    # These compatibility copies retain the exact paths bound by the public
    # campaign and auxiliary locks.  Only small manifests/configs are copied;
    # probability arrays and checkpoints remain external hash-bound artifacts.
    (
        "outputs/binary_train/fives/clipseg/seed-0/train_config.json",
        "outputs/binary_train/fives/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/fives/deeplabv3/seed-0/train_config.json",
        "outputs/binary_train/fives/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/isic/clipseg/seed-0/train_config.json",
        "outputs/binary_train/isic/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/isic/deeplabv3/seed-0/train_config.json",
        "outputs/binary_train/isic/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/kvasir/clipseg/seed-0/train_config.json",
        "outputs/binary_train/kvasir/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/kvasir/deeplabv3/seed-0/train_config.json",
        "outputs/binary_train/kvasir/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/pet/clipseg/seed-0/train_config.json",
        "outputs/binary_train/pet/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/pet/deeplabv3/seed-0/train_config.json",
        "outputs/binary_train/pet/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/tn3k/clipseg/seed-0/train_config.json",
        "outputs/binary_train/tn3k/clipseg/seed-0/train_config.json",
    ),
    (
        "outputs/binary_train/tn3k/deeplabv3/seed-0/train_config.json",
        "outputs/binary_train/tn3k/deeplabv3/seed-0/train_config.json",
    ),
    (
        "outputs/binary_artifacts/pet/clipseg-general/e0d454ce916c5ca9/manifest.json",
        "outputs/binary_artifacts/pet/clipseg-general/e0d454ce916c5ca9/manifest.json",
    ),
    (
        "outputs/binary_artifacts/pet/clipseg-target/7261edbb5f763435/manifest.json",
        "outputs/binary_artifacts/pet/clipseg-target/7261edbb5f763435/manifest.json",
    ),
    (
        "outputs/binary_artifacts/pet/deeplabv3-target/ced4dc6e67bc178f/manifest.json",
        "outputs/binary_artifacts/pet/deeplabv3-target/ced4dc6e67bc178f/manifest.json",
    ),
    (
        "outputs/binary_artifacts/pet/deeplabv3-external/640ef98bcd028dc3/manifest.json",
        "outputs/binary_artifacts/pet/deeplabv3-external/640ef98bcd028dc3/manifest.json",
    ),
    (
        "outputs/binary_artifacts/kvasir/clipseg-general/eb7304bdd945035d/manifest.json",
        "outputs/binary_artifacts/kvasir/clipseg-general/eb7304bdd945035d/manifest.json",
    ),
    (
        "outputs/binary_artifacts/kvasir/clipseg-target/7ce9cddd0f9704fd/manifest.json",
        "outputs/binary_artifacts/kvasir/clipseg-target/7ce9cddd0f9704fd/manifest.json",
    ),
    (
        "outputs/binary_artifacts/kvasir/deeplabv3-target/be7dedef6990d54c/manifest.json",
        "outputs/binary_artifacts/kvasir/deeplabv3-target/be7dedef6990d54c/manifest.json",
    ),
    (
        "outputs/binary_artifacts/fives/clipseg-general/cb5c08e73819574c/manifest.json",
        "outputs/binary_artifacts/fives/clipseg-general/cb5c08e73819574c/manifest.json",
    ),
    (
        "outputs/binary_artifacts/fives/clipseg-target/ef8e1b5176b92b87/manifest.json",
        "outputs/binary_artifacts/fives/clipseg-target/ef8e1b5176b92b87/manifest.json",
    ),
    (
        "outputs/binary_artifacts/fives/deeplabv3-target/e7e1e90e13ce3dfa/manifest.json",
        "outputs/binary_artifacts/fives/deeplabv3-target/e7e1e90e13ce3dfa/manifest.json",
    ),
    (
        "outputs/binary_artifacts/isic/clipseg-general/3e57aa259264c1be/manifest.json",
        "outputs/binary_artifacts/isic/clipseg-general/3e57aa259264c1be/manifest.json",
    ),
    (
        "outputs/binary_artifacts/isic/clipseg-target/a5d30aa0ec8b0742/manifest.json",
        "outputs/binary_artifacts/isic/clipseg-target/a5d30aa0ec8b0742/manifest.json",
    ),
    (
        "outputs/binary_artifacts/isic/deeplabv3-target/48cf75646c61ecf4/manifest.json",
        "outputs/binary_artifacts/isic/deeplabv3-target/48cf75646c61ecf4/manifest.json",
    ),
    (
        "outputs/binary_artifacts/tn3k/clipseg-general/9626e945ef94b1e5/manifest.json",
        "outputs/binary_artifacts/tn3k/clipseg-general/9626e945ef94b1e5/manifest.json",
    ),
    (
        "outputs/binary_artifacts/tn3k/clipseg-target/603dc94883b5f395/manifest.json",
        "outputs/binary_artifacts/tn3k/clipseg-target/603dc94883b5f395/manifest.json",
    ),
    (
        "outputs/binary_artifacts/tn3k/deeplabv3-target/efce266864cd5269/manifest.json",
        "outputs/binary_artifacts/tn3k/deeplabv3-target/efce266864cd5269/manifest.json",
    ),
    (
        "outputs/binary_assembled/fives/clipseg-general/74db5fc8c0672236/manifest.json",
        "results/assembled/fives/clipseg-general/74db5fc8c0672236/manifest.json",
    ),
    (
        "outputs/binary_assembled/fives/clipseg-general/74db5fc8c0672236/records.jsonl",
        "results/assembled/fives/clipseg-general/74db5fc8c0672236/records.jsonl",
    ),
    (
        "outputs/binary_assembled/fives/clipseg-target/627922c22aa3aa21/manifest.json",
        "results/assembled/fives/clipseg-target/627922c22aa3aa21/manifest.json",
    ),
    (
        "outputs/binary_assembled/fives/clipseg-target/627922c22aa3aa21/records.jsonl",
        "results/assembled/fives/clipseg-target/627922c22aa3aa21/records.jsonl",
    ),
    (
        "outputs/binary_assembled/fives/deeplabv3-target/a62d754378787146/manifest.json",
        "results/assembled/fives/deeplabv3-target/a62d754378787146/manifest.json",
    ),
    (
        "outputs/binary_assembled/fives/deeplabv3-target/a62d754378787146/records.jsonl",
        "results/assembled/fives/deeplabv3-target/a62d754378787146/records.jsonl",
    ),
    (
        "outputs/binary_assembled/isic/clipseg-general/500bf2dd2f3564b2/manifest.json",
        "results/assembled/isic/clipseg-general/500bf2dd2f3564b2/manifest.json",
    ),
    (
        "outputs/binary_assembled/isic/clipseg-general/500bf2dd2f3564b2/records.jsonl",
        "results/assembled/isic/clipseg-general/500bf2dd2f3564b2/records.jsonl",
    ),
    (
        "outputs/binary_assembled/isic/clipseg-target/69fc681ff737f148/manifest.json",
        "results/assembled/isic/clipseg-target/69fc681ff737f148/manifest.json",
    ),
    (
        "outputs/binary_assembled/isic/clipseg-target/69fc681ff737f148/records.jsonl",
        "results/assembled/isic/clipseg-target/69fc681ff737f148/records.jsonl",
    ),
    (
        "outputs/binary_assembled/isic/deeplabv3-target/a42288ea78a340b1/manifest.json",
        "results/assembled/isic/deeplabv3-target/a42288ea78a340b1/manifest.json",
    ),
    (
        "outputs/binary_assembled/isic/deeplabv3-target/a42288ea78a340b1/records.jsonl",
        "results/assembled/isic/deeplabv3-target/a42288ea78a340b1/records.jsonl",
    ),
    (
        "outputs/binary_assembled/kvasir/clipseg-general/8f5786edfc8d5993/manifest.json",
        "results/assembled/kvasir/clipseg-general/8f5786edfc8d5993/manifest.json",
    ),
    (
        "outputs/binary_assembled/kvasir/clipseg-general/8f5786edfc8d5993/records.jsonl",
        "results/assembled/kvasir/clipseg-general/8f5786edfc8d5993/records.jsonl",
    ),
    (
        "outputs/binary_assembled/kvasir/clipseg-target/1a1c02b968ab86a4/manifest.json",
        "results/assembled/kvasir/clipseg-target/1a1c02b968ab86a4/manifest.json",
    ),
    (
        "outputs/binary_assembled/kvasir/clipseg-target/1a1c02b968ab86a4/records.jsonl",
        "results/assembled/kvasir/clipseg-target/1a1c02b968ab86a4/records.jsonl",
    ),
    (
        "outputs/binary_assembled/kvasir/deeplabv3-target/5cca940975b7f02d/manifest.json",
        "results/assembled/kvasir/deeplabv3-target/5cca940975b7f02d/manifest.json",
    ),
    (
        "outputs/binary_assembled/kvasir/deeplabv3-target/5cca940975b7f02d/records.jsonl",
        "results/assembled/kvasir/deeplabv3-target/5cca940975b7f02d/records.jsonl",
    ),
    (
        "outputs/binary_assembled/pet/clipseg-general/88d2033138fc971a/manifest.json",
        "results/assembled/pet/clipseg-general/88d2033138fc971a/manifest.json",
    ),
    (
        "outputs/binary_assembled/pet/clipseg-general/88d2033138fc971a/records.jsonl",
        "results/assembled/pet/clipseg-general/88d2033138fc971a/records.jsonl",
    ),
    (
        "outputs/binary_assembled/pet/clipseg-target/cd4341aaed2bda20/manifest.json",
        "results/assembled/pet/clipseg-target/cd4341aaed2bda20/manifest.json",
    ),
    (
        "outputs/binary_assembled/pet/clipseg-target/cd4341aaed2bda20/records.jsonl",
        "results/assembled/pet/clipseg-target/cd4341aaed2bda20/records.jsonl",
    ),
    (
        "outputs/binary_assembled/pet/deeplabv3-external/014bfe3ad787f51e/manifest.json",
        "results/assembled/pet/deeplabv3-external/014bfe3ad787f51e/manifest.json",
    ),
    (
        "outputs/binary_assembled/pet/deeplabv3-external/014bfe3ad787f51e/records.jsonl",
        "results/assembled/pet/deeplabv3-external/014bfe3ad787f51e/records.jsonl",
    ),
    (
        "outputs/binary_assembled/pet/deeplabv3-target/fd2f61c609c18fba/manifest.json",
        "results/assembled/pet/deeplabv3-target/fd2f61c609c18fba/manifest.json",
    ),
    (
        "outputs/binary_assembled/pet/deeplabv3-target/fd2f61c609c18fba/records.jsonl",
        "results/assembled/pet/deeplabv3-target/fd2f61c609c18fba/records.jsonl",
    ),
    (
        "outputs/binary_assembled/tn3k/clipseg-general/4faed2465c790f8c/manifest.json",
        "results/assembled/tn3k/clipseg-general/4faed2465c790f8c/manifest.json",
    ),
    (
        "outputs/binary_assembled/tn3k/clipseg-general/4faed2465c790f8c/records.jsonl",
        "results/assembled/tn3k/clipseg-general/4faed2465c790f8c/records.jsonl",
    ),
    (
        "outputs/binary_assembled/tn3k/clipseg-target/eccb8f4f045a5473/manifest.json",
        "results/assembled/tn3k/clipseg-target/eccb8f4f045a5473/manifest.json",
    ),
    (
        "outputs/binary_assembled/tn3k/clipseg-target/eccb8f4f045a5473/records.jsonl",
        "results/assembled/tn3k/clipseg-target/eccb8f4f045a5473/records.jsonl",
    ),
    (
        "outputs/binary_assembled/tn3k/deeplabv3-target/01b4c3a58986cf27/manifest.json",
        "results/assembled/tn3k/deeplabv3-target/01b4c3a58986cf27/manifest.json",
    ),
    (
        "outputs/binary_assembled/tn3k/deeplabv3-target/01b4c3a58986cf27/records.jsonl",
        "results/assembled/tn3k/deeplabv3-target/01b4c3a58986cf27/records.jsonl",
    ),
)


@dataclass(frozen=True)
class CopyItem:
    source_root: Path
    source: Path
    destination: Path


@dataclass(frozen=True)
class SyncTarget:
    name: str
    root: Path
    items: tuple[CopyItem, ...]


@dataclass(frozen=True)
class _PublicationGroup:
    name: str
    payloads: tuple[CopyItem, ...]
    manifest: CopyItem
    guard: CopyItem

    @property
    def items(self) -> tuple[CopyItem, ...]:
        return self.payloads + (self.manifest, self.guard)


def _safe_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe allowlist path: {path!r}")
    return relative


def _item(source_root: Path, source: str, target_root: Path, target: str) -> CopyItem:
    source_rel = _safe_relative(source)
    target_rel = _safe_relative(target)
    return CopyItem(
        source_root,
        source_root.joinpath(*source_rel.parts),
        target_root.joinpath(*target_rel.parts),
    )


def _validated_source_path(source: Path, source_root: Path) -> Path:
    """Return a regular source path contained by a non-symlink trust root."""

    absolute_root = Path(os.path.abspath(source_root))
    absolute_source = Path(os.path.abspath(source))
    if absolute_root.is_symlink() or not absolute_root.is_dir():
        raise RuntimeError(
            f"allowlisted source root is missing or a symlink: {absolute_root}"
        )
    try:
        absolute_source.relative_to(absolute_root)
    except ValueError as error:
        raise RuntimeError(
            f"allowlisted source is outside its source root: {absolute_source}"
        ) from error

    for ancestor in absolute_source.parents:
        if ancestor.is_symlink():
            raise RuntimeError(f"allowlisted source ancestor is a symlink: {ancestor}")
    if absolute_source.is_symlink() or not absolute_source.is_file():
        raise RuntimeError(
            f"allowlisted source is missing or a symlink: {absolute_source}"
        )

    resolved_root = absolute_root.resolve(strict=True)
    resolved_source = absolute_source.resolve(strict=True)
    if not resolved_source.is_relative_to(resolved_root):
        raise RuntimeError(
            f"resolved allowlisted source escapes its source root: {absolute_source}"
        )
    return absolute_source


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest(value, *, location: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{location} must be a SHA-256 hex digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{location} must be a SHA-256 hex digest")
    return normalized


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_generated_manifest(path: Path, *, artifact_type: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"generated manifest is not a regular file: {path}")
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot load generated manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"generated manifest must contain one object: {path}")
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"generated manifest schema is unsupported: {path}")
    if manifest.get("artifact_type") != artifact_type:
        raise RuntimeError(f"generated manifest artifact_type is unsupported: {path}")
    return manifest


def _validate_output_hashes(
    docs_root: Path,
    outputs,
    *,
    expected_names: set[str],
    location: str,
) -> None:
    if not isinstance(outputs, list):
        raise RuntimeError(f"{location}.outputs must be a list")
    names = [item.get("path") for item in outputs if isinstance(item, dict)]
    if len(names) != len(outputs) or len(names) != len(set(names)):
        raise RuntimeError(f"{location}.outputs must contain unique objects")
    if set(names) != expected_names:
        raise RuntimeError(f"{location}.outputs differs from the public file set")
    for item in outputs:
        expected_sha = _digest(
            item.get("sha256"), location=f"{location}.outputs[{item['path']}].sha256"
        )
        source = docs_root / "Figures" / item["path"]
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"generated output is not a regular file: {source}")
        if _sha256_file(source) != expected_sha:
            raise RuntimeError(f"generated output SHA-256 mismatch: {source}")


def _validate_risk_coverage_closure(docs_root: Path) -> None:
    sentinel = docs_root / OPTIONAL_GENERATED_FIGURE_SENTINEL_FILES[0]
    manifest_path = docs_root / OPTIONAL_GENERATED_FIGURE_MANIFEST_FILES[0]
    manifest = _load_generated_manifest(
        manifest_path,
        artifact_type="selectseg.risk_coverage_render_manifest",
    )
    render_spec = manifest.get("render_spec")
    if not isinstance(render_spec, dict):
        raise RuntimeError("risk render manifest render_spec must be an object")
    render_spec_sha = _digest(
        manifest.get("render_spec_sha256"),
        location="risk render manifest.render_spec_sha256",
    )
    if _canonical_sha256(render_spec) != render_spec_sha:
        raise RuntimeError("risk render-spec SHA-256 mismatch")
    source_bundle = render_spec.get("source_artifact_bundle")
    if not isinstance(source_bundle, dict):
        raise RuntimeError("risk render spec source bundle must be an object")
    source_bundle_sha = _digest(
        render_spec.get("source_artifact_bundle_sha256"),
        location="risk render spec.source_artifact_bundle_sha256",
    )
    if _canonical_sha256(source_bundle) != source_bundle_sha:
        raise RuntimeError("risk source-artifact-bundle SHA-256 mismatch")
    _digest(
        render_spec.get("plot_source_sha256"),
        location="risk render spec.plot_source_sha256",
    )
    expected_names = {
        Path(path).name for path in OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES
    }
    if render_spec.get("outputs") != [
        Path(path).name for path in OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES
    ]:
        raise RuntimeError(
            "risk render spec output order differs from the public file set"
        )
    _validate_output_hashes(
        docs_root,
        manifest.get("outputs"),
        expected_names=expected_names,
        location="risk render manifest",
    )
    sentinel_text = sentinel.read_text(encoding="utf-8")
    manifest_sha = _sha256_file(manifest_path)
    for digest in (render_spec_sha, source_bundle_sha, manifest_sha):
        if digest not in sentinel_text:
            raise RuntimeError(
                "risk completion sentinel does not bind the render closure"
            )
    for output_name in expected_names:
        if (
            render_spec_sha.encode("ascii")
            not in (docs_root / "Figures" / output_name).read_bytes()
        ):
            raise RuntimeError(
                f"risk PDF metadata does not bind the render spec: {output_name}"
            )


def _validate_qualitative_closure(docs_root: Path) -> None:
    manifest_path = docs_root / "Figures/qualitative_manifest.json"
    manifest = _load_generated_manifest(
        manifest_path,
        artifact_type="selectseg.binary_qualitative_manuscript",
    )
    expected_names = {
        Path(path).name
        for path in OPTIONAL_GENERATED_QUALITATIVE_FILES
        if not path.endswith("qualitative_manifest.json")
    }
    _validate_output_hashes(
        docs_root,
        manifest.get("outputs"),
        expected_names=expected_names,
        location="qualitative manifest",
    )
    selection_sha = _digest(
        manifest.get("selection_sha256"),
        location="qualitative manifest.selection_sha256",
    )
    renderer_source_sha = _digest(
        manifest.get("renderer_source_sha256"),
        location="qualitative manifest.renderer_source_sha256",
    )
    expected_render_id = hashlib.sha256(
        (selection_sha + "\0" + renderer_source_sha).encode("ascii")
    ).hexdigest()[:16]
    if manifest.get("render_id") != expected_render_id:
        raise RuntimeError("qualitative manifest render_id is inconsistent")
    required_digests = (
        selection_sha,
        _digest(
            manifest.get("campaign_lock_sha256"),
            location="qualitative manifest.campaign_lock_sha256",
        ),
        renderer_source_sha,
        _digest(
            manifest.get("source_render_manifest_sha256"),
            location="qualitative manifest.source_render_manifest_sha256",
        ),
    )
    tex = (docs_root / "Figures/qualitative_cases.tex").read_text(encoding="utf-8")
    if any(digest not in tex for digest in required_digests):
        raise RuntimeError("qualitative TeX does not bind the published source closure")


def _validate_public_seed_result_closure(repo_root: Path) -> None:
    """Validate the exact path-free seed release and all guard bindings."""

    # Imports stay local: the ordinary manuscript sync does not need to import
    # the experiment stack when the optional seed release is absent.
    from scripts import export_binary_seed_provenance as seed_exporter

    analysis_path = _validated_source_path(
        repo_root / _PUBLIC_SEED_ANALYSIS_SOURCE, repo_root
    )
    scheduler_path = _validated_source_path(
        repo_root / _PUBLIC_SEED_SCHEDULER_SOURCE, repo_root
    )
    provenance_path = _validated_source_path(
        repo_root / _PUBLIC_SEED_GUARD_SOURCE, repo_root
    )
    seed_exporter.load_public_seed_release(
        analysis_path,
        scheduler_path,
        provenance_path,
    )


def _validate_public_seed_replay_closure(repo_root: Path) -> None:
    """Validate the 30-condition path-free replay and its write-last guard."""

    from scripts.replay_seed_robustness import verify_replay_payloads

    guard_path = _validated_source_path(
        repo_root / _PUBLIC_SEED_REPLAY_GUARD_SOURCE, repo_root
    )
    guard = _load_generated_manifest(
        guard_path, artifact_type="selectseg.portable_seed_replay_complete"
    )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "lock_sha256",
        "condition_count",
        "portable_file_count",
        "portable_bundle_sha256",
    }
    if set(guard) != expected_fields:
        raise RuntimeError("portable seed replay guard schema is invalid")
    if guard["condition_count"] != 30 or guard["portable_file_count"] != 60:
        raise RuntimeError("portable seed replay guard has an invalid grid")
    lock_path = _validated_source_path(
        repo_root / _PUBLIC_SEED_REPLAY_LOCK_SOURCE, repo_root
    )
    if _sha256_file(lock_path) != _digest(
        guard["lock_sha256"], location="seed replay guard.lock_sha256"
    ):
        raise RuntimeError("portable seed replay lock differs from its guard")

    bundle_digest = hashlib.sha256()
    prefix = "outputs/public_seed/seed_records/"
    for source, _ in OPTIONAL_PUBLIC_SEED_REPLAY_PAYLOAD_FILES:
        path = _validated_source_path(repo_root / source, repo_root)
        if not source.startswith(prefix):
            raise RuntimeError("portable seed replay source is noncanonical")
        bundle_digest.update(source.removeprefix(prefix).encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(hashlib.sha256(path.read_bytes()).digest())
        bundle_digest.update(b"\0")
    if bundle_digest.hexdigest() != _digest(
        guard["portable_bundle_sha256"],
        location="seed replay guard.portable_bundle_sha256",
    ):
        raise RuntimeError("portable seed replay bundle digest mismatch")

    destination_to_source = {
        destination: source for source, destination in OPTIONAL_PUBLIC_SEED_REPLAY_FILES
    }
    destination_to_source.update(
        {
            "results/seed_robustness_analysis.json": _PUBLIC_SEED_ANALYSIS_SOURCE,
            "tables/seed_robustness.tex": "docs/Tables/seed_robustness.tex",
            "tables/seed_sensitivity_main.tex": (
                "docs/Tables/seed_sensitivity_main.tex"
            ),
        }
    )

    def read_replay_payload(relative: str) -> bytes:
        try:
            source = destination_to_source[relative]
        except KeyError as error:
            raise RuntimeError(
                f"seed replay requests a non-public source: {relative}"
            ) from error
        return _validated_source_path(repo_root / source, repo_root).read_bytes()

    _, report = verify_replay_payloads(lock_path.read_bytes(), read_replay_payload)
    if report.get("condition_count") != 30 or report.get("seed_count") != 3:
        raise RuntimeError("portable seed replay closure has an invalid grid")


def _publication_specs() -> tuple[
    tuple[
        str,
        tuple[str, ...],
        str,
        str,
        Callable[[Path], None],
    ],
    ...,
]:
    qualitative_manifest = "Figures/qualitative_manifest.json"
    qualitative_guard = "Figures/qualitative_cases.tex"
    return (
        (
            "risk-coverage",
            OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES,
            OPTIONAL_GENERATED_FIGURE_MANIFEST_FILES[0],
            OPTIONAL_GENERATED_FIGURE_SENTINEL_FILES[0],
            _validate_risk_coverage_closure,
        ),
        (
            "qualitative",
            tuple(
                path
                for path in OPTIONAL_GENERATED_QUALITATIVE_FILES
                if path not in {qualitative_manifest, qualitative_guard}
            ),
            qualitative_manifest,
            qualitative_guard,
            _validate_qualitative_closure,
        ),
        (
            "public-seed-results",
            (_PUBLIC_SEED_ANALYSIS_SOURCE,),
            _PUBLIC_SEED_SCHEDULER_SOURCE,
            _PUBLIC_SEED_GUARD_SOURCE,
            _validate_public_seed_result_closure,
        ),
        (
            "public-seed-replay",
            tuple(
                source for source, _ in OPTIONAL_PUBLIC_SEED_REPLAY_PAYLOAD_FILES
            ),
            _PUBLIC_SEED_REPLAY_LOCK_SOURCE,
            _PUBLIC_SEED_REPLAY_GUARD_SOURCE,
            _validate_public_seed_replay_closure,
        ),
    )


def _validated_publication_groups(target: SyncTarget) -> tuple[_PublicationGroup, ...]:
    """Resolve only exact, closure-validated generated groups for guarded writes."""

    relative_sources: list[tuple[str, CopyItem]] = []
    for item in target.items:
        source = Path(os.path.abspath(item.source))
        source_root = Path(os.path.abspath(item.source_root))
        try:
            relative = source.relative_to(source_root).as_posix()
        except ValueError:
            continue
        relative_sources.append((relative, item))

    groups = []
    for name, payloads, manifest, guard, closure_validator in _publication_specs():
        expected = payloads + (manifest, guard)
        matching = [(path, item) for path, item in relative_sources if path in expected]
        if not matching:
            continue
        by_path: dict[str, CopyItem] = {}
        for path, item in matching:
            if path in by_path:
                raise RuntimeError(f"duplicate {name} publication source: {path}")
            by_path[path] = item
        if set(by_path) != set(expected):
            missing = ", ".join(path for path in expected if path not in by_path)
            raise RuntimeError(f"{name} publication group is incomplete: {missing}")

        source_roots = {
            Path(os.path.abspath(item.source_root)) for item in by_path.values()
        }
        if len(source_roots) != 1:
            raise RuntimeError(f"{name} publication group crosses source roots")
        source_root = source_roots.pop()
        for item in by_path.values():
            _validated_source_path(item.source, source_root)
        if name in {"public-seed-results", "public-seed-replay"} and (
            target.name != "github" or source_root != Path(os.path.abspath(REPO_ROOT))
        ):
            raise RuntimeError(
                "public seed results are restricted to the GitHub mirror"
            )

        try:
            destination_by_source = {
                path: item.destination.relative_to(target.root).as_posix()
                for path, item in by_path.items()
            }
        except ValueError as error:
            raise RuntimeError(
                f"{name} publication destination escapes target root"
            ) from error
        direct_layout = {path: path for path in expected}
        github_layout = {path: f"docs/{path}" for path in expected}
        allowed_layouts = (direct_layout, github_layout)
        if name == "public-seed-results":
            allowed_layouts = (dict(OPTIONAL_PUBLIC_SEED_RESULT_FILES),)
        elif name == "public-seed-replay":
            allowed_layouts = (dict(OPTIONAL_PUBLIC_SEED_REPLAY_FILES),)
        if destination_by_source not in allowed_layouts:
            raise RuntimeError(f"{name} publication destinations are not allowlisted")

        closure_validator(source_root)
        groups.append(
            _PublicationGroup(
                name=name,
                payloads=tuple(by_path[path] for path in payloads),
                manifest=by_path[manifest],
                guard=by_path[guard],
            )
        )
    return tuple(groups)


def _validated_optional_generated_group(
    docs_root: Path,
    paths: tuple[str, ...],
    *,
    label: str,
    closure_validator,
) -> tuple[str, ...]:
    """Admit an optional generated group only as one complete, valid closure."""

    candidates = tuple(docs_root / path for path in paths)
    if not any(path.exists() or path.is_symlink() for path in candidates):
        return ()
    incomplete = tuple(
        path
        for path, candidate in zip(paths, candidates, strict=True)
        if not candidate.is_file() or candidate.is_symlink()
    )
    if incomplete:
        missing = ", ".join(incomplete)
        raise RuntimeError(
            f"{label} generated group is incomplete or non-regular: {missing}"
        )
    for candidate in candidates:
        _validated_source_path(candidate, docs_root)
    closure_validator(docs_root)
    return paths


def _validated_optional_public_seed_results(
    repo_root: Path,
) -> tuple[tuple[str, str], ...]:
    """Admit only the exact seed release selected by its write-last guard.

    The analysis and scheduler summary are legitimate intermediate products:
    the scheduler summary exists before any downstream seed analysis can run.
    Until the provenance guard is present, treat those payloads as unpublished
    and leave every mirror destination unmanaged.  Once the guard appears,
    fail closed unless all three regular files form the strict release closure.
    """

    candidates = tuple(
        repo_root / source for source, _ in OPTIONAL_PUBLIC_SEED_RESULT_FILES
    )
    guard = repo_root / _PUBLIC_SEED_GUARD_SOURCE
    if not guard.exists() and not guard.is_symlink():
        return ()
    incomplete = tuple(
        source
        for (source, _), candidate in zip(
            OPTIONAL_PUBLIC_SEED_RESULT_FILES, candidates, strict=True
        )
        if not candidate.is_file() or candidate.is_symlink()
    )
    if incomplete:
        missing = ", ".join(incomplete)
        raise RuntimeError(
            "public seed result group is incomplete or non-regular: " + missing
        )
    for candidate in candidates:
        _validated_source_path(candidate, repo_root)
    _validate_public_seed_result_closure(repo_root)
    return OPTIONAL_PUBLIC_SEED_RESULT_FILES


def _validated_optional_public_seed_replay(
    repo_root: Path,
) -> tuple[tuple[str, str], ...]:
    """Admit only a complete replay tree selected by its write-last guard."""

    guard = repo_root / _PUBLIC_SEED_REPLAY_GUARD_SOURCE
    if not guard.exists() and not guard.is_symlink():
        return ()
    candidates = tuple(repo_root / source for source, _ in OPTIONAL_PUBLIC_SEED_REPLAY_FILES)
    incomplete = tuple(
        source
        for (source, _), candidate in zip(
            OPTIONAL_PUBLIC_SEED_REPLAY_FILES, candidates, strict=True
        )
        if not candidate.is_file() or candidate.is_symlink()
    )
    if incomplete:
        raise RuntimeError(
            "public seed replay group is incomplete or non-regular: "
            + ", ".join(incomplete)
        )
    for candidate in candidates:
        _validated_source_path(candidate, repo_root)
    _validate_public_seed_replay_closure(repo_root)
    return OPTIONAL_PUBLIC_SEED_REPLAY_FILES


def _targets() -> dict[str, SyncTarget]:
    docs_root = REPO_ROOT / "docs"
    overleaf_root = REPO_ROOT / "overleaf"
    github_root = REPO_ROOT / "github"

    # Unit-sized publication tests construct manuscript-only roots.  The
    # checked-in main root lock is the write-last guard that activates this
    # closure validator; a real GitHub sync always has it, while a deliberately
    # minimal paper-only fixture does not.
    scientific_guard = REPO_ROOT / GITHUB_SCIENTIFIC_ROOT_CONFIGS[0][1]
    if scientific_guard.exists() or scientific_guard.is_symlink():
        _validate_public_scientific_input_closure(REPO_ROOT)

    risk_coverage_files = _validated_optional_generated_group(
        docs_root,
        OPTIONAL_GENERATED_MANUSCRIPT_PDF_FILES
        + OPTIONAL_GENERATED_FIGURE_SENTINEL_FILES
        + OPTIONAL_GENERATED_FIGURE_MANIFEST_FILES,
        label="risk-coverage",
        closure_validator=_validate_risk_coverage_closure,
    )
    qualitative_files = _validated_optional_generated_group(
        docs_root,
        OPTIONAL_GENERATED_QUALITATIVE_FILES,
        label="qualitative",
        closure_validator=_validate_qualitative_closure,
    )
    public_seed_results = _validated_optional_public_seed_results(REPO_ROOT)
    public_seed_replay = _validated_optional_public_seed_replay(REPO_ROOT)

    optional_manuscript_files = (
        OPTIONAL_GENERATED_MANUSCRIPT_FILES + qualitative_files + risk_coverage_files
    )
    manuscript_paths = MANUSCRIPT_FILES + tuple(
        path for path in optional_manuscript_files if (docs_root / path).is_file()
    )
    overleaf_items = tuple(
        _item(docs_root, path, overleaf_root, path) for path in manuscript_paths
    )
    github_paths = (
        GITHUB_ROOT_FILES
        + GITHUB_PACKAGE_FILES
        + GITHUB_SCRIPT_FILES
        + GITHUB_SCIENTIFIC_INPUT_FILES
        + GITHUB_TEST_FILES
    )
    github_items = (
        tuple(_item(REPO_ROOT, path, github_root, path) for path in github_paths)
        + tuple(
            _item(docs_root, path, github_root, f"docs/{path}")
            for path in manuscript_paths
        )
        + tuple(
            _item(REPO_ROOT, source, github_root, target)
            for source, target in GITHUB_RESULT_FILES
        )
        + tuple(
            _item(REPO_ROOT, source, github_root, target)
            for source, target in public_seed_results
        )
        + tuple(
            _item(REPO_ROOT, source, github_root, target)
            for source, target in public_seed_replay
        )
    )
    return {
        "overleaf": SyncTarget("overleaf", overleaf_root, overleaf_items),
        "github": SyncTarget("github", github_root, github_items),
    }


def _verify_git_clone(target: SyncTarget) -> Path:
    if target.root.is_symlink() or not (target.root / ".git").is_dir():
        raise RuntimeError(f"{target.root} is not a non-symlink Git clone")
    result = subprocess.run(
        ["git", "-C", str(target.root), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = Path(result.stdout.strip()).resolve()
    expected = target.root.resolve()
    if actual != expected:
        raise RuntimeError(
            f"refusing to sync {target.name}: Git root is {actual}, expected {expected}"
        )
    return expected


def _contains_secret_marker(content: bytes) -> bool:
    # Construct the markers so the synchronizer's own source does not contain
    # the literal credential prefixes it rejects.
    markers = (
        b"olp" + b"_",
        b"ghp" + b"_",
        b"github_pat" + b"_",
        b"sk" + b"-proj-",
        b"-----BEGIN " + b"PRIVATE KEY-----",
    )
    return any(marker in content for marker in markers)


def _iter_json_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_strings(child)
    elif isinstance(value, str):
        yield value


def _validate_portable_scientific_json(content: bytes, *, source: str) -> dict:
    """Reject private paths/identities in one public scientific seal."""

    if _contains_secret_marker(content):
        raise RuntimeError(f"possible credential marker in scientific seal: {source}")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"scientific seal is not strict UTF-8 JSON: {source}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"scientific seal must be a JSON object: {source}")
    private_markers = (
        "scratch.global",
        "zhan9381",
        "sinianzhang",
        "sinian zhang",
    )
    for value in _iter_json_strings(payload):
        lowered = value.lower()
        is_windows_absolute = (
            len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}
        )
        if (
            value.startswith(("/", "~"))
            or is_windows_absolute
            or "://" in value
            or value.startswith("git@")
            or any(marker in lowered for marker in private_markers)
        ):
            raise RuntimeError(
                f"scientific seal contains a private or non-portable value: {source}"
            )
    return payload


def _validate_public_scientific_input_closure(repo_root: Path) -> None:
    """Validate both root locks and their exact portable publication closure."""

    declared = set(GITHUB_SCIENTIFIC_INPUT_FILES)
    root_paths = {root for _, root in GITHUB_SCIENTIFIC_ROOT_CONFIGS}
    component_paths = declared - root_paths
    referenced_components: set[str] = set()

    for relative in GITHUB_SCIENTIFIC_INPUT_FILES:
        source = _validated_source_path(repo_root / relative, repo_root)
        _validate_portable_scientific_json(source.read_bytes(), source=relative)

    for config_relative, root_relative in GITHUB_SCIENTIFIC_ROOT_CONFIGS:
        if config_relative not in GITHUB_SCRIPT_FILES or root_relative not in declared:
            raise RuntimeError("scientific config/root pair is not publicly allowlisted")
        config_path = _validated_source_path(repo_root / config_relative, repo_root)
        config = json.loads(config_path.read_bytes())
        root_binding = config.get("scientific_input_lock")
        if not isinstance(root_binding, dict) or root_binding.get("path") != root_relative:
            raise RuntimeError(
                f"campaign config does not bind its public root lock: {config_relative}"
            )
        expected_sha256 = root_binding.get("sha256")
        loaded = load_root_lock(
            root_relative,
            repo_root=repo_root,
            expected_sha256=expected_sha256,
            verify_component_manifests=True,
        )
        components = loaded["lock"]["components"]
        referenced_components.update(
            binding["path"] for binding in components["datasets"]
        )
        referenced_components.update(
            components[name]["path"]
            for name in ("source", "base_models", "checkpoints", "environment")
        )

    if referenced_components != component_paths:
        missing = sorted(referenced_components - component_paths)
        unbound = sorted(component_paths - referenced_components)
        raise RuntimeError(
            "public scientific component closure differs from its allowlist: "
            f"missing={missing}, unbound={unbound}"
        )


def _validated_content(item: CopyItem, target_root: Path) -> bytes:
    source = _validated_source_path(item.source, item.source_root)
    destination = item.destination
    if destination.is_symlink():
        raise RuntimeError(f"refusing to replace destination symlink: {destination}")
    if not destination.resolve(strict=False).is_relative_to(target_root):
        raise RuntimeError(f"destination escapes target clone: {destination}")
    content = source.read_bytes()
    if _contains_secret_marker(content):
        raise RuntimeError(
            f"possible credential marker in allowlisted source: {item.source}"
        )
    return content


def _atomic_write(destination: Path, content: bytes, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.sync-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _withdraw_publication_guard(group: _PublicationGroup, target_root: Path) -> None:
    """Remove one exact validated guard before changing its published closure."""

    destination = group.guard.destination
    if destination.is_symlink():
        raise RuntimeError(
            f"refusing to remove publication guard symlink: {destination}"
        )
    if not destination.resolve(strict=False).is_relative_to(target_root):
        raise RuntimeError(f"publication guard escapes target clone: {destination}")
    if not destination.exists():
        return
    if not destination.is_file():
        raise RuntimeError(f"publication guard is not a regular file: {destination}")
    destination.unlink()
    _fsync_directory(destination.parent)


def sync(target: SyncTarget, *, apply: bool) -> tuple[int, int]:
    target_root = _verify_git_clone(target)
    changes: list[tuple[CopyItem, bytes]] = []
    validated_contents: dict[CopyItem, bytes] = {}
    unchanged = 0
    seen_destinations: set[Path] = set()

    for item in target.items:
        resolved_destination = item.destination.resolve(strict=False)
        if resolved_destination in seen_destinations:
            raise RuntimeError(f"duplicate allowlist destination: {item.destination}")
        seen_destinations.add(resolved_destination)
        content = _validated_content(item, target_root)
        validated_contents[item] = content
        if item.destination.is_file() and item.destination.read_bytes() == content:
            unchanged += 1
        else:
            changes.append((item, content))

    action = "WRITE" if apply else "WOULD WRITE"
    print(f"[{target.name}] {len(changes)} change(s), {unchanged} unchanged")
    for item, _ in changes:
        relative = item.destination.relative_to(target.root).as_posix()
        print(f"  {action} {relative}")

    publication_groups = _validated_publication_groups(target)
    changed_items = {item for item, _ in changes}
    active_groups = tuple(
        group
        for group in publication_groups
        if any(item in changed_items for item in group.items)
    )

    def publish(item: CopyItem) -> None:
        content = validated_contents[item]
        source_mode = item.source.stat().st_mode & 0o777
        _atomic_write(item.destination, content, source_mode)

    if apply:
        for group in active_groups:
            _withdraw_publication_guard(group, target_root)
        for group in active_groups:
            for item in group.payloads:
                publish(item)
        for group in active_groups:
            publish(group.manifest)
        for group in active_groups:
            publish(group.guard)

        active_items = {item for group in active_groups for item in group.items}
        for item, _ in changes:
            if item in active_items:
                continue
            publish(item)
    print(f"[{target.name}] unmanaged destination files were preserved")
    return len(changes), unchanged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", "overleaf", "github"),
        default="all",
        help="local clone to inspect or update",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform writes (the default is a dry run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _targets()
    selected = targets.values() if args.target == "all" else (targets[args.target],)
    changed = 0
    for target in selected:
        target_changes, _ = sync(target, apply=args.apply)
        changed += target_changes
    if not args.apply and changed:
        print(f"dry run complete: {changed} allowlisted file(s) would change")


if __name__ == "__main__":
    main()
