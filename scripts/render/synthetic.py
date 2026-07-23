"""Render compact TeX and PDF summaries from synthetic posterior analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from selectseg.studies.synthetic import (
    COUPLINGS,
    LOSSES,
    SHARPNESS_LEVELS,
    _load_json,
    dice_loss,
)
from selectseg.studies.synthetic_matrix import ESTIMATOR_COUPLINGS
from selectseg.geometry import binary_surface, normalized_penalized_boundary_losses
from selectseg.quadrature import sha256_file


COUPLING_LABELS = {
    "shared_threshold": "Shared threshold",
    "independent_bernoulli": "Independent Bernoulli",
    "local_block_threshold": "Local block threshold",
    "bimodal_antithetic": "Bimodal antithetic",
}
LOSS_LABELS = {"dice": "Dice", "nhd": "HD", "nhd95": "HD95"}
COLORS = {
    "shared_threshold": "#1f77b4",
    "independent_bernoulli": "#d62728",
    "local_block_threshold": "#2ca02c",
    "bimodal_antithetic": "#9467bd",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("posterior-summary", "coupling-matrix", "hd95-contamination"),
        default="posterior-summary",
    )
    parser.add_argument("--analysis")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.mode != "hd95-contamination" and args.analysis is None:
        parser.error("--analysis is required in this mode")
    if args.mode == "hd95-contamination" and args.analysis is not None:
        parser.error("--analysis is not used in hd95-contamination mode")
    return args


def _validated_analysis(path):
    analysis_path, analysis = _load_json(path, name="synthetic analysis")
    required = {
        "analysis_schema_version",
        "analysis_id",
        "created_utc",
        "campaign_id",
        "mode",
        "lock",
        "num_cells",
        "source_manifests",
        "pilot_gate",
        "groups",
        "headline_groups",
        "coupling_summaries",
        "interpretation",
    }
    if set(analysis) != required or analysis.get("analysis_schema_version") != 1:
        raise ValueError("synthetic analysis has an unexpected schema")
    expected = 12 if analysis["mode"] == "pilot" else 360
    if (
        analysis["num_cells"] != expected
        or len(analysis["source_manifests"]) != expected
    ):
        raise ValueError("synthetic analysis is incomplete")
    if {row["coupling"] for row in analysis["coupling_summaries"]} != set(COUPLINGS):
        raise ValueError("synthetic analysis lacks a coupling summary")
    return analysis_path, analysis


def _format(value):
    if value is None:
        return "--"
    if abs(value) < 0.00005:
        return r"$<\!10^{-4}$"
    return f"{value:.4f}"


def render_tex(analysis, source_sha):
    by_coupling = {row["coupling"]: row for row in analysis["coupling_summaries"]}
    lines = [
        f"% Auto-generated from synthetic analysis SHA-256: {source_sha}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Known-posterior stress test. Each pair reports mean absolute "
        r"M32 working-risk error and mean AURC regret (lower is better), aggregated "
        r"over the completed synthetic cells. AURC regret is multiplied by 100 for "
        r"display only.}",
        r"\label{tab:synthetic-posterior}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Dice} & \multicolumn{2}{c}{HD} & "
        r"\multicolumn{2}{c}{HD95} \\",
        r"Coupling & $|\widetilde r-r^\star|$ & $100\times$AURC reg. & "
        r"$|\widetilde r-r^\star|$ & $100\times$AURC reg. & "
        r"$|\widetilde r-r^\star|$ & $100\times$AURC reg. \\",
        r"\midrule",
    ]
    for coupling in COUPLINGS:
        row = by_coupling[coupling]
        values = []
        for loss in LOSSES:
            estimator = row["losses"][loss]["estimators"]["m32"]
            values.extend(
                [
                    _format(estimator["mean_absolute_score_error"]),
                    _format(100.0 * estimator["mean_aurc_regret"]),
                ]
            )
        lines.append(COUPLING_LABELS[coupling] + " & " + " & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize",
            r"True conditional risks are Monte Carlo estimates; their posterior-draw "
            r"Monte Carlo SE is reported in the machine-readable analysis. Scalar loss-pushforward "
            r"$W_1$ is reported for all losses, but no HD95 mask-Wasserstein corollary "
            r"is claimed. M128 is a numerical reference for boundary losses; Dice-Exact "
            r"is the only exact threshold integral.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_figure(analysis, output, source_sha):
    lookup = {
        (row["coupling"], row["sharpness"]): row for row in analysis["headline_groups"]
    }
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), sharex=True)
    x = range(len(SHARPNESS_LEVELS))
    for column, loss in enumerate(LOSSES):
        for coupling in COUPLINGS:
            rows = [lookup[(coupling, level)] for level in SHARPNESS_LEVELS]
            error = [
                row["losses"][loss]["estimators"]["m32"]["mean_absolute_score_error"]
                for row in rows
            ]
            regret = [
                100.0
                * row["losses"][loss]["estimators"]["m32"]["mean_aurc_regret"]
                for row in rows
            ]
            axes[0, column].plot(
                x,
                error,
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                color=COLORS[coupling],
                label=COUPLING_LABELS[coupling],
            )
            axes[1, column].plot(
                x,
                regret,
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                color=COLORS[coupling],
            )
        axes[0, column].set_title(LOSS_LABELS[loss])
        axes[1, column].set_xticks(list(x), SHARPNESS_LEVELS)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Mean absolute score error")
    axes[1, 0].set_ylabel("Mean AURC regret (×100)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.subplots_adjust(top=0.84, bottom=0.12, left=0.09, right=0.99, wspace=0.28)
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Known-posterior synthetic stress test",
            "Subject": f"Source analysis SHA-256 {source_sha}",
            "Creator": "scripts.render.synthetic",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def hd95_contamination_curve():
    """Construct a deterministic mask family crossing the pooled 5% tail."""

    shape = (192, 256)
    reference = np.zeros(shape, dtype=bool)
    reference[60:132, 80:176] = True
    reference_surface_size = int(binary_surface(reference).sum())
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    eligible = (
        ~reference
        & ((yy % 3) == 0)
        & ((xx % 3) == 0)
        & ~((yy >= 57) & (yy < 135) & (xx >= 77) & (xx < 179))
    )
    # Add the most remote isolated pixels first.  Their pairwise spacing keeps
    # each pixel on the candidate surface and makes the pooled contamination
    # fraction exactly k / (2 |surface(reference)| + k).
    distance_proxy = np.hypot(yy - 96, xx - 128)
    coordinates = np.argwhere(eligible)
    order = np.argsort(
        distance_proxy[eligible], kind="stable"
    )[::-1]
    coordinates = coordinates[order]
    target_percent = np.asarray([0.0, 1.0, 4.0, 4.9, 5.0, 5.1, 6.0, 10.0])
    rows = []
    diagonal = math.hypot(*shape)
    for percentage in target_percent:
        fraction = percentage / 100.0
        outliers = (
            0
            if fraction == 0
            else int(round(2 * reference_surface_size * fraction / (1 - fraction)))
        )
        candidate = reference.copy()
        selected = coordinates[:outliers]
        candidate[selected[:, 0], selected[:, 1]] = True
        candidate_surface_size = int(binary_surface(candidate).sum())
        pooled_size = reference_surface_size + candidate_surface_size
        observed_fraction = (candidate_surface_size - reference_surface_size) / pooled_size
        boundary = normalized_penalized_boundary_losses(reference, candidate)
        rows.append(
            {
                "target_contamination_percent": float(percentage),
                "observed_contamination_percent": float(100 * observed_fraction),
                "outlier_surface_pixels": outliers,
                "dice_loss": dice_loss(reference, candidate),
                "hd_pixels": boundary.nhd * diagonal,
                "hd95_pixels": boundary.nhd95 * diagonal,
            }
        )
    return rows


def render_hd95_contamination(output: Path, data_output: Path) -> None:
    """Show robustness below 5% and the percentile transition above it."""

    rows = hd95_contamination_curve()
    x = np.asarray([row["observed_contamination_percent"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.15))
    axes[0].plot(
        x,
        [row["hd_pixels"] for row in rows],
        marker="o",
        label="HD",
        color="#9b2c2c",
    )
    axes[0].plot(
        x,
        [row["hd95_pixels"] for row in rows],
        marker="s",
        label="HD95",
        color="#1f5a94",
    )
    axes[0].axvline(5, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Boundary distance (pixels)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        x,
        [100 * row["dice_loss"] for row in rows],
        marker="o",
        color="#2d7d46",
    )
    axes[1].axvline(5, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Dice loss (×100)")
    for axis in axes:
        axis.set_xlabel("Remote surface contamination (%)")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "HD95 is robust below its 5% tail, then changes rapidly",
        fontsize=10.5,
    )
    figure.tight_layout()
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "HD95 contamination transition",
            "Creator": "scripts.render.synthetic",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    data_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "construction": (
                    "fixed rectangle plus mutually separated remote one-pixel "
                    "components; pooled surface percentile uses NumPy defaults"
                ),
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validated_matrix_analysis(path):
    analysis_path, analysis = _load_json(path, name="synthetic matrix analysis")
    required = {
        "schema_version",
        "analysis_id",
        "created_utc",
        "campaign_id",
        "mode",
        "config",
        "num_cells",
        "source_manifests",
        "true_coupling_summaries",
        "sharpness_summaries",
        "interpretation",
    }
    if set(analysis) != required or analysis.get("schema_version") != 1:
        raise ValueError("synthetic matrix analysis has an unexpected schema")
    expected = 12 if analysis["mode"] == "pilot" else 360
    if analysis["num_cells"] != expected:
        raise ValueError("synthetic matrix analysis is incomplete")
    truths = {row["true_coupling"] for row in analysis["true_coupling_summaries"]}
    if truths != set(COUPLINGS):
        raise ValueError("synthetic matrix lacks a true-coupling summary")
    return analysis_path, analysis


def render_matrix_tex(analysis, source_sha):
    lookup = {
        row["true_coupling"]: row
        for row in analysis["true_coupling_summaries"]
    }
    estimator_labels = {
        "shared_threshold": "Shared threshold",
        "independent_bernoulli": "Independent Bernoulli",
        "local_block_threshold": "Local block",
        "spatial_copula": "Spatial copula",
    }
    lines = [
        f"% Auto-generated from coupling-matrix analysis SHA-256: {source_sha}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Known-posterior coupling matrix. Estimator couplings are rows "
        r"and true couplings are columns. Each cell is mean absolute conditional-"
        r"risk error / AURC regret $\times100$ (lower is better). Pixel marginals, "
        r"actions, cohorts, and Monte Carlo budgets are fixed.}",
        r"\label{tab:synthetic-coupling-matrix}",
    ]
    for loss in LOSSES:
        lines.extend(
            [
                rf"\textbf{{{LOSS_LABELS[loss]}}}\par\vspace{{1pt}}",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
                r"\toprule",
                "Estimator $Q$ & "
                + " & ".join(COUPLING_LABELS[value] for value in COUPLINGS)
                + r" \\",
                r"\midrule",
            ]
        )
        values = {}
        for estimator in ESTIMATOR_COUPLINGS:
            values[estimator] = []
            for truth in COUPLINGS:
                entry = lookup[truth]["losses"][loss]["estimators"][estimator]
                values[estimator].append(
                    (
                        entry["mean_absolute_risk_error"],
                        100 * entry["mean_aurc_regret"],
                    )
                )
        minima = [
            min(values[estimator][column][0] for estimator in ESTIMATOR_COUPLINGS)
            for column in range(len(COUPLINGS))
        ]
        for estimator in ESTIMATOR_COUPLINGS:
            cells = []
            for column, (error, regret) in enumerate(values[estimator]):
                cell = f"{error:.4f} / {regret:.3f}"
                if error == minima[column]:
                    cell = rf"\bestresult{{{cell}}}"
                cells.append(cell)
            lines.append(estimator_labels[estimator] + " & " + " & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular*}", r"\par\smallskip"])
    lines.extend([r"\end{table*}", ""])
    return "\n".join(lines)


def render_matrix_figure(analysis, output, source_sha):
    lookup = {
        row["true_coupling"]: row
        for row in analysis["true_coupling_summaries"]
    }
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 6.0))
    short_truth = ["Shared", "Independent", "Local block", "Bimodal"]
    short_estimator = ["Shared", "Independent", "Local block", "Copula"]
    for column, loss in enumerate(LOSSES):
        error = np.asarray(
            [
                [
                    lookup[truth]["losses"][loss]["estimators"][estimator][
                        "mean_absolute_risk_error"
                    ]
                    for estimator in ESTIMATOR_COUPLINGS
                ]
                for truth in COUPLINGS
            ]
        )
        regret = np.asarray(
            [
                [
                    100
                    * lookup[truth]["losses"][loss]["estimators"][estimator][
                        "mean_aurc_regret"
                    ]
                    for estimator in ESTIMATOR_COUPLINGS
                ]
                for truth in COUPLINGS
            ]
        )
        for row, (matrix, label) in enumerate(
            ((error, "Mean absolute risk error"), (regret, "AURC regret ×100"))
        ):
            image = axes[row, column].imshow(matrix, cmap="magma_r", aspect="auto")
            axes[row, column].set_xticks(range(4), short_estimator, rotation=35, ha="right")
            axes[row, column].set_yticks(range(4), short_truth)
            if column == 0:
                axes[row, column].set_ylabel("True coupling\n" + label)
            for y in range(4):
                for x in range(4):
                    axes[row, column].text(
                        x,
                        y,
                        f"{matrix[y, x]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if matrix[y, x] > np.nanmedian(matrix) else "black",
                    )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
        axes[0, column].set_title(LOSS_LABELS[loss])
    figure.suptitle("Estimator coupling × true coupling", fontsize=11)
    figure.tight_layout()
    figure.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Known-posterior coupling matrix",
            "Subject": f"Source analysis SHA-256 {source_sha}",
            "Creator": "scripts.render.synthetic",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.mode == "hd95-contamination":
        figure_path = output_dir / "hd95_contamination.pdf"
        data_path = output_dir / "hd95_contamination.json"
        if any(path.exists() or path.is_symlink() for path in (figure_path, data_path)):
            raise FileExistsError("refusing to overwrite HD95 contamination output")
        output_dir.mkdir(parents=True, exist_ok=True)
        render_hd95_contamination(figure_path, data_path)
        print(figure_path)
        print(data_path)
        return
    if args.mode == "coupling-matrix":
        analysis_path, analysis = _validated_matrix_analysis(args.analysis)
        source_sha = sha256_file(analysis_path)
        table_path = output_dir / "synthetic_coupling_matrix.tex"
        figure_path = output_dir / "synthetic_coupling_matrix.pdf"
        manifest_path = output_dir / "synthetic_coupling_matrix_render.manifest.json"
        if any(
            path.exists() or path.is_symlink()
            for path in (table_path, figure_path, manifest_path)
        ):
            raise FileExistsError("refusing to overwrite coupling-matrix render")
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path.write_text(
            render_matrix_tex(analysis, source_sha), encoding="utf-8"
        )
        render_matrix_figure(analysis, figure_path, source_sha)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_analysis": {
                        "path": str(analysis_path),
                        "sha256": source_sha,
                    },
                    "outputs": [
                        {"path": str(table_path), "sha256": sha256_file(table_path)},
                        {"path": str(figure_path), "sha256": sha256_file(figure_path)},
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(table_path)
        print(figure_path)
        return
    analysis_path, analysis = _validated_analysis(args.analysis)
    source_sha = sha256_file(analysis_path)
    tex_path = output_dir / "synthetic_posterior_summary.tex"
    figure_path = output_dir / "synthetic_posterior_summary.pdf"
    manifest_path = output_dir / "synthetic_posterior_render.manifest.json"
    for path in (tex_path, figure_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite rendered artifact: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(render_tex(analysis, source_sha), encoding="utf-8")
    render_figure(analysis, figure_path, source_sha)
    manifest = {
        "render_schema_version": 1,
        "source_analysis": {"path": str(analysis_path), "sha256": source_sha},
        "outputs": [
            {"path": str(tex_path), "sha256": sha256_file(tex_path)},
            {"path": str(figure_path), "sha256": sha256_file(figure_path)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(tex_path)
    print(figure_path)


if __name__ == "__main__":
    main()
