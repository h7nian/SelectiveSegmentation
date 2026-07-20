"""Render compact TeX and PDF summaries from synthetic posterior analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selectseg.synthetic_posterior import (
    COUPLINGS,
    LOSSES,
    SHARPNESS_LEVELS,
    _load_json,
)
from selectseg.threshold_estimators import sha256_file


COUPLING_LABELS = {
    "shared_threshold": "Shared threshold",
    "independent_bernoulli": "Independent Bernoulli",
    "local_block_threshold": "Local block threshold",
    "bimodal_antithetic": "Bimodal antithetic",
}
LOSS_LABELS = {"dice": "Dice", "nhd": "nHD", "nhd95": "nHD95"}
COLORS = {
    "shared_threshold": "#1f77b4",
    "independent_bernoulli": "#d62728",
    "local_block_threshold": "#2ca02c",
    "bimodal_antithetic": "#9467bd",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


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
        r" & \multicolumn{2}{c}{Dice} & \multicolumn{2}{c}{nHD} & "
        r"\multicolumn{2}{c}{nHD95} \\",
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
            "Creator": "scripts.render_synthetic_posterior",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    analysis_path, analysis = _validated_analysis(args.analysis)
    source_sha = sha256_file(analysis_path)
    output_dir = Path(args.output_dir)
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
